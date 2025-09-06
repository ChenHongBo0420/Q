# Copyright 2021 The Commplax Authors.
# Modified: Split-loss + Gradient Routing for invariant (back-end) vs equivariant (front-end)
# Drop-in replacement: same public API (train/test/equalize_dataset); add test_ext for GMI/NGMI.

from jax import numpy as jnp, random, jit, value_and_grad, lax
import flax
from flax.core import freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
from flax import linen as nn
from commplax import util, comm, cxopt, op, optim
from commplax.module import core, layer
import numpy as np
from functools import partial
from collections import namedtuple
from tqdm.auto import tqdm
from typing import Any, Optional, Union
import jax

# -------------------- Types --------------------
Model = namedtuple('Model', 'module initvar overlaps name')
Array = Any
Dict  = Union[dict, flax.core.FrozenDict]

# -------------------- Defaults（可按需改，不改也能跑） --------------------
# 辅损（bit-BCE）目标权重 alpha 在 train() 里热启动到该目标
DEFAULT_ALPHA_TARGET   = 1.0     # 建议 0.8~1.0
DEFAULT_WARMUP_STEPS   = 500     # 前 warmup_steps 线性从 0→alpha_target
DEFAULT_BACK_LR_MULT   = 2.5     # 末端参数上对辅损梯度放大（1.5~3.0）
ETA_PHASE              = 0.5     # 在辅损/边距损里移除 ETA*phi* 的全局相位（温和去相位）
TAU_BCE                = 1.6     # bit-BCE 温度，>1 更平滑
DEFAULT_PHASE_W        = 2e-3    # 仅 FOEAf 的相位相干损权重（5e-4~5e-3）
DEFAULT_MARGIN_W       = 0.08    # 末端 margin 副损权重（0.02~0.1）
DEFAULT_MARGIN_M       = 0.25    # 最近-次近星座间隔 margin

# back-end 路由命中的模块名（辅损只更新这些；不要包含 FOEAf）
_BACK_KEYS = ('RConv','MIMOAF','CDC','Demap','demap')

# -------------------- Base module（与你现有一致） --------------------
def make_base_module(steps: int = 3,
                     dtaps: int = 261,
                     ntaps: int = 41,
                     rtaps: int = 61,
                     init_fn: tuple = (core.delta, core.gauss),
                     w0 = 0.,
                     mode: str = 'train'):
    _assert_taps(dtaps, ntaps, rtaps)
    d_init, n_init = init_fn
    if mode == 'train':
        mimo_train = True
    elif mode == 'test':
        mimo_train = cxopt.piecewise_constant([200000], [True, False])
    else:
        raise ValueError('invalid mode %s' % mode)
    base = layer.Serial(
        layer.FDBP(steps=steps, dtaps=dtaps, ntaps=ntaps, d_init=d_init, n_init=n_init),
        layer.BatchPowerNorm(mode=mode),
        layer.MIMOFOEAf(name='FOEAf', w0=w0, train=mimo_train,
                        preslicer=core.conv1d_slicer(rtaps), foekwargs={}),
        layer.vmap(layer.Conv1d)(name='RConv', taps=rtaps),
        layer.MIMOAF(train=mimo_train)
    )
    return base

def _assert_taps(dtaps, ntaps, rtaps, sps=2):
    assert dtaps % sps, f'dtaps must be odd number, got {dtaps}'
    assert ntaps % sps, f'ntaps must be odd number, got {ntaps}'
    assert rtaps % sps, f'rtaps must be odd number, got {rtaps}'

def fdbp_init(a: dict, xi: float = 1.1, steps: Optional[int] = None):
    def d_init(key, shape, dtype=jnp.complex64):
        dtaps = shape[0]
        d0, _ = comm.dbp_params(a['samplerate'], a['distance']/a['spans'], a['spans'],
                                dtaps, a['lpdbm'] - 3, virtual_spans=steps)
        return d0[0, :, 0]
    def n_init(key, shape, dtype=jnp.float32):
        dtaps = shape[0]
        _, n0 = comm.dbp_params(a['samplerate'], a['distance']/a['spans'], a['spans'],
                                dtaps, a['lpdbm'] - 3, virtual_spans=steps)
        return xi * n0[0, 0, 0] * core.gauss(key, shape, dtype)
    return d_init, n_init

def model_init(data,
               base_conf: dict,
               sparams_flatkeys: list,
               n_symbols: int = 4000,
               sps : int = 2,
               name='Model'):
    mod = make_base_module(**base_conf, w0=data.w0)
    y0 = data.y[:n_symbols * sps]
    rng0 = random.PRNGKey(0)
    z0, v0 = mod.init(rng0, core.Signal(y0))
    ol = z0.t.start - z0.t.stop
    sparams, params = util.dict_split(v0['params'], sparams_flatkeys)
    state = v0['af_state']; aux = v0['aux_inputs']; const = v0['const']
    return Model(mod, (params, state, aux, const, sparams), ol, name)

# -------------------- helpers --------------------
def energy(x): return jnp.sum(jnp.square(x))

# 16QAM 常量（与你们口径一致）
CONST_16QAM = jnp.array([
    -3-3j, -3-1j, -3+3j, -3+1j,
    -1-3j, -1-1j, -1+3j, -1+1j,
     3-3j,  3-1j,  3+3j,  3+1j,
     1-3j,  1-1j,  1+3j,  1+1j
], dtype=jnp.complex64) / jnp.sqrt(10.)
_bits = (
    (0,0,0,0), (0,0,0,1), (0,0,1,0), (0,0,1,1),
    (0,1,0,0), (0,1,0,1), (0,1,1,0), (0,1,1,1),
    (1,0,0,0), (1,0,0,1), (1,0,1,0), (1,0,1,1),
    (1,1,0,0), (1,1,0,1), (1,1,1,0), (1,1,1,1)
)
BIT_MAP      = jnp.array(_bits, dtype=jnp.float32)             # [16,4]
BIT_WEIGHTS  = jnp.array([1.2, 1.0, 1.0, 0.8], dtype=jnp.float32)

# -------------------- 掩膜（梯度路由） --------------------
def make_back_mask(params: Dict) -> Dict:
    flat = flatten_dict(unfreeze(params))
    mflat = {}
    for path, leaf in flat.items():
        key = '/'.join([str(p) for p in path])
        hit = any(tag in key for tag in _BACK_KEYS)
        mflat[path] = jnp.ones_like(leaf) if hit else jnp.zeros_like(leaf)
    return freeze(unflatten_dict(mflat))

def make_foe_mask(params: Dict) -> Dict:
    flat = flatten_dict(unfreeze(params))
    mflat = {}
    for path, leaf in flat.items():
        key = '/'.join([str(p) for p in path])
        hit = ('FOEAf' in key)
        mflat[path] = jnp.ones_like(leaf) if hit else jnp.zeros_like(leaf)
    return freeze(unflatten_dict(mflat))

# -------------------- 损失函数（主/三种副损） --------------------
def loss_main_fn(module, params, state, y, x, aux, const, sparams, λ_kl: float = 1e-4):
    """主损：SI-SNR + 0.1*EVM + 轻正则（不做相位去除 → 等变，留梯度给 F/LDBP）"""
    params_net = util.dict_merge(params, sparams)
    z_out, state_new = module.apply(
        {'params': params_net, 'aux_inputs': aux, 'const': const, **state},
        core.Signal(y)
    )
    x_ref = x[z_out.t.start:z_out.t.stop]
    yhat  = z_out.val

    evm = jnp.mean(jnp.abs(yhat - x_ref)**2) / (jnp.mean(jnp.abs(x_ref)**2) + 1e-8)

    t = x_ref.reshape(-1); e = yhat.reshape(-1)
    a = jnp.vdot(t, e) / (jnp.vdot(t, t) + 1e-8)
    s = a * t
    err = e - s
    si_snr_neg = -10.0 * jnp.log10(
        (jnp.real(jnp.vdot(s, s)) + 1e-8) / (jnp.real(jnp.vdot(err, err)) + 1e-8)
    )
    reg = 0.5 * jnp.mean(jnp.square(jnp.abs(yhat)))
    loss = si_snr_neg + 0.1 * evm + λ_kl * reg
    return loss, state_new

def loss_aux_fn(module, params, state, y, x, aux, const, sparams, tau: float = TAU_BCE):
    """副损1（不变）：温和去相位后的 16QAM bit-BCE（只路由到末端）"""
    params_net = util.dict_merge(params, sparams)
    z_out, _ = module.apply({'params': params_net, 'aux_inputs': aux, 'const': const, **state},
                            core.Signal(y))
    x_ref = x[z_out.t.start:z_out.t.stop]
    yhat  = z_out.val

    # 温和去相位（估计 phi*，去 ETA_PHASE·phi*）
    TAIL_WIN = 16384
    T  = int(yhat.shape[0]); W = min(TAIL_WIN, T)
    ys = yhat[-W:].reshape(-1); xs = x_ref[-W:].reshape(-1)
    p  = lax.stop_gradient(jnp.vdot(ys, xs) / (jnp.abs(jnp.vdot(ys, xs)) + 1e-8))
    y_al = yhat * jnp.power(jnp.conj(p), ETA_PHASE)

    logits = -jnp.square(jnp.abs(y_al[..., None] - CONST_16QAM)) / (tau * tau)
    logp   = logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True)
    probs  = jnp.exp(logp)
    p1     = (probs @ BIT_MAP); p0 = 1.0 - p1
    idx    = jnp.argmin(jnp.square(jnp.abs(x_ref[..., None] - CONST_16QAM)), axis=-1)
    bits   = BIT_MAP[idx]
    bce    = -(bits * jnp.log(p1 + 1e-12) + (1. - bits) * jnp.log(p0 + 1e-12))
    loss   = (bce * BIT_WEIGHTS).mean()
    return loss

def loss_phase_foe_fn(module, params, state, y, x, aux, const, sparams):
    """副损2：仅对 FOEAf 的相位相干（1 - |corr|）"""
    params_net = util.dict_merge(params, sparams)
    z_out, _ = module.apply({'params': params_net, 'aux_inputs': aux, 'const': const, **state},
                            core.Signal(y))
    x_ref = x[z_out.t.start:z_out.t.stop]
    yhat  = z_out.val
    TAIL_WIN = 16384
    T  = int(yhat.shape[0]); W = min(TAIL_WIN, T)
    ys = yhat[-W:].reshape(-1); xs = x_ref[-W:].reshape(-1)
    num = jnp.abs(jnp.vdot(ys, xs))
    den = jnp.linalg.norm(ys) * jnp.linalg.norm(xs) + 1e-8
    return 1.0 - num / den

def loss_margin_fn(module, params, state, y, x, aux, const, sparams, margin_m: float):
    """副损3：末端 margin（最近-次近星座距离要拉开）"""
    params_net = util.dict_merge(params, sparams)
    z_out, _ = module.apply({'params': params_net, 'aux_inputs': aux, 'const': const, **state},
                            core.Signal(y))
    x_ref = x[z_out.t.start:z_out.t.stop]
    yhat  = z_out.val

    # 复用温和相位去除
    TAIL_WIN = 16384
    T  = int(yhat.shape[0]); W = min(TAIL_WIN, T)
    ys = yhat[-W:].reshape(-1); xs = x_ref[-W:].reshape(-1)
    p  = lax.stop_gradient(jnp.vdot(ys, xs) / (jnp.abs(jnp.vdot(ys, xs)) + 1e-8))
    y_al = yhat * jnp.power(jnp.conj(p), ETA_PHASE)

    d2 = jnp.square(jnp.abs(y_al[..., None] - CONST_16QAM))  # [N,16]
    d_sorted = jnp.sort(d2, axis=-1)
    d1, d2n  = d_sorted[..., 0], d_sorted[..., 1]
    return jnp.maximum(0.0, margin_m + d1 - d2n).mean()

# -------------------- Update step（梯度路由总线） --------------------
@partial(jit, backend='cpu', static_argnums=(0, 1))
def update_step(module: layer.Layer,
                opt: cxopt.Optimizer,
                i: int,
                opt_state: tuple,
                module_state: Dict,
                y: Array,
                x: Array,
                aux: Dict,
                const: Dict,
                sparams: Dict,
                alpha_aux: float,
                mask_back: Dict,
                mask_foe: Dict,
                back_lr_mult: float,
                phase_w: float,
                margin_w: float,
                margin_m: float):
    params = opt.params_fn(opt_state)

    # 主损
    (Lm, state_main), g_main = value_and_grad(
        loss_main_fn, argnums=1, has_aux=True)(
        module, params, module_state, y, x, aux, const, sparams
    )

    # 副损1：bit-BCE（末端）
    La, g_aux = value_and_grad(
        lambda p: loss_aux_fn(module, p, module_state, y, x, aux, const, sparams)
    )(params)

    # 副损2：FOEAf 相位相干（仅 FOEAf）
    Lphi, g_phi = value_and_grad(
        lambda p: loss_phase_foe_fn(module, p, module_state, y, x, aux, const, sparams)
    )(params)
    g_phi_routed = jax.tree_map(lambda gp, mf: phase_w * gp * mf, g_phi, mask_foe)

    # 副损3：末端 margin
    Lmg, g_mg = value_and_grad(
        lambda p: loss_margin_fn(module, p, module_state, y, x, aux, const, sparams, margin_m)
    )(params)
    g_mg_routed = jax.tree_map(lambda gm, mb: margin_w * gm * mb, g_mg, mask_back)

    # 先把两路结构化副损加到主梯度（都已路由）
    g_sum = jax.tree_map(lambda gm, a, b: gm + a + b, g_main, g_phi_routed, g_mg_routed)

    # 再叠加“末端 bit-BCE 副损”，并放大到 back_lr_mult
    g_routed = jax.tree_map(lambda gm, ga, mb: gm + alpha_aux * ga * mb * back_lr_mult,
                            g_sum, g_aux, mask_back)

    opt_state = opt.update_fn(i, g_routed, opt_state)
    Ltot = Lm + alpha_aux * La + phase_w * Lphi + margin_w * Lmg
    return (Lm, La, Lphi, Lmg, Ltot), opt_state, state_main

# -------------------- Data batching（保持不变） --------------------
def get_train_batch(ds, batchsize: int, overlaps: int, sps: int = 2):
    flen = batchsize + overlaps
    fstep = batchsize
    ds_y = op.frame_gen(ds.y, flen * sps, fstep * sps)
    ds_x = op.frame_gen(ds.x, flen, fstep)
    n_batches = op.frame_shape(ds.x.shape, flen, fstep)[0]
    return n_batches, zip(ds_y, ds_x)

# -------------------- train（热启动 + 路由 + 增强副损；外部 API 兼容） --------------------
def train(model: Model,
          data,
          batch_size: int = 500,
          n_iter = None,
          alpha_aux: float = None,         # 兼容老参；若给出则覆盖 alpha_target
          alpha_target: float = DEFAULT_ALPHA_TARGET,
          warmup_steps: int = DEFAULT_WARMUP_STEPS,
          back_lr_mult: float = DEFAULT_BACK_LR_MULT,
          phase_loss_weight: float = DEFAULT_PHASE_W,
          margin_w: float = DEFAULT_MARGIN_W,
          margin_m: float = DEFAULT_MARGIN_M,
          opt: optim.Optimizer = optim.adam(optim.piecewise_constant([500, 1000], [1e-4, 1e-5, 1e-6]))):
    params, module_state, aux, const, sparams = model.initvar
    opt_state = opt.init_fn(params)

    # 兼容：若外部传了 alpha_aux，则当作目标值
    if alpha_aux is not None:
        alpha_target = alpha_aux

    mask_back = make_back_mask(params)  # 末端掩膜
    mask_foe  = make_foe_mask(params)   # FOEAf 掩膜

    n_batch, batch_gen = get_train_batch(data, batch_size, model.overlaps)
    n_iter = n_batch if n_iter is None else min(n_iter, n_batch)

    for i, (y, x) in tqdm(enumerate(batch_gen), total=n_iter, desc='training', leave=False):
        if i >= n_iter: break
        aux = core.dict_replace(aux, {'truth': x})

        # 热启动：前 warmup_steps 线性从 0 → alpha_target
        alpha_t = float(alpha_target) * float(min(1.0, i / max(1, warmup_steps)))

        (Lm, La, Lphi, Lmg, Ltot), opt_state, module_state = update_step(
            model.module, opt, i, opt_state, module_state,
            y, x, aux, const, sparams,
            alpha_t, mask_back, mask_foe,
            back_lr_mult, phase_loss_weight, margin_w, margin_m
        )
        # 对外接口保持：yield 总损 / 最新参数 / 模块状态
        yield float(Ltot), opt.params_fn(opt_state), module_state

# -------------------- test（保持不变；默认旧指标） --------------------
def test(model: Model,
         params: Dict,
         data,
         eval_range: tuple=(300000, -20000),
         metric_fn=comm.qamqot):
    state, aux, const, sparams = model.initvar[1:]
    aux = core.dict_replace(aux, {'truth': data.x})
    if params is None:
        params = model.initvar[0]
    z, _ = jit(model.module.apply, backend='cpu')({
        'params': util.dict_merge(params, sparams),
        'aux_inputs': aux, 'const': const, **state
    }, core.Signal(data.y))
    metric = metric_fn(z.val, data.x[z.t.start:z.t.stop],
                       scale=np.sqrt(10), eval_range=eval_range)
    return metric, z

# -------------------- test_ext（便捷：直接给 GMI/NGMI/C4D） --------------------
def test_ext(model: Model,
             params: Dict,
             data,
             L: int = 16,
             d4_dims: int = 2,
             eval_range: tuple=(300000, -20000)):
    """和 test 一样的接口，但 metric 固定为扩展指标（含 GMI/NGMI/C4D/Gap）。"""
    state, aux, const, sparams = model.initvar[1:]
    aux = core.dict_replace(aux, {'truth': data.x})
    if params is None:
        params = model.initvar[0]
    z, _ = jit(model.module.apply, backend='cpu')({
        'params': util.dict_merge(params, sparams),
        'aux_inputs': aux, 'const': const, **state
    }, core.Signal(data.y))
    metric = comm.qamqot_ext(z.val, data.x[z.t.start:z.t.stop],
                             L=L, d4_dims=d4_dims,
                             scale=np.sqrt(10), eval_range=eval_range)
    return metric, z

# -------------------- equalize_dataset（保持不变） --------------------
def equalize_dataset(model_te, params, state_bundle, data):
    module_state, aux, const, sparams = state_bundle
    z,_ = jax.jit(model_te.module.apply, backend='cpu')(
        {'params': util.dict_merge(params, sparams),
         'aux_inputs': aux, 'const': const, **module_state},
        core.Signal(data.y))
    start, stop = z.t.start, z.t.stop
    z_eq  = np.asarray(z.val[:,0])
    s_ref = np.asarray(data.x)[start:stop,0]
    return z_eq, s_ref

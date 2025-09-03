# =========================================
# SOTANN1 with Quotient-Space + Gradient Routing (full)
# 保留原结构；新增：商空间VJP、相位一致性、梯度后路由、裁剪、护栏
# =========================================
from typing import Any, Optional, Union, Tuple, List
from collections import namedtuple
from functools import partial

import numpy as np
import jax
from jax import numpy as jnp, random, jit, value_and_grad
import flax
from flax import linen as nn
from flax.core import freeze
from flax.traverse_util import flatten_dict, unflatten_dict

from commplax import util, comm, cxopt, op, optim
from commplax.module import core, layer

Array = Any
Dict  = Union[dict, flax.core.FrozenDict]
Model = namedtuple('Model', 'module initvar overlaps name')

# ===========================
# 可调小超参
# ===========================
ROUTE_KEEP_PHASE   = 0.35   # 自定义VJP里保留的相位分量比例（0~1）
ROUTE_MASK_GUARD   = 2048   # 相位估计中心窗护边
BCE_TEMP           = 0.90   # Bit-BCE温度(<1更锐)
ETA_PHASE          = 0.05   # 相位一致性损失权重（给FDBP相位等变牵引）
BETA_CE            = 1.2    # 分类头权重
LAMBDA_KL          = 2e-5   # 信息瓶颈（能量）正则
CLIP_NORM          = 2.5    # 全局梯度裁剪
TF_ENABLE          = True   # 是否 teacher forcing 把 truth 喂到 aux

# ===========================
# 16QAM + Bit-BCE
# ===========================
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
BIT_MAP     = jnp.array(_bits, dtype=jnp.float32)          # [16,4]
BIT_WEIGHTS = jnp.array([1.2, 1.0, 1.0, 0.8], jnp.float32) # [4]

def _bit_bce_loss_16qam(pred_sym: Array, true_sym: Array) -> Array:
    ps = pred_sym.reshape(-1)
    ts = true_sym.reshape(-1)
    logits = -jnp.square(jnp.abs(ps[..., None] - CONST_16QAM)) / BCE_TEMP
    logp   = logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True)
    probs  = jnp.exp(logp)
    p1     = (probs @ BIT_MAP)              # [N,4]
    p0     = 1.0 - p1
    idx    = jnp.argmin(jnp.square(jnp.abs(ts[..., None] - CONST_16QAM)), axis=-1)
    bits   = BIT_MAP[idx]
    bce    = -(bits*jnp.log(p1+1e-12) + (1.-bits)*jnp.log(p0+1e-12))
    return (bce * BIT_WEIGHTS).mean()

# ===========================
# 形状护栏：y/x 对齐
# ===========================
def _match_y_x(yhat: jnp.ndarray, x_ref: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    def _squeeze_last1(a):
        return a[..., 0] if (a.ndim >= 2 and a.shape[-1] == 1) else a
    yv = _squeeze_last1(yhat)
    xv = _squeeze_last1(x_ref)
    L = min(yv.shape[0], xv.shape[0])
    yv, xv = yv[:L], xv[:L]
    # 通道对齐
    if yv.ndim == 1 and xv.ndim == 1:
        return yv, xv
    if yv.ndim == 2 and xv.ndim == 1:
        xv = jnp.repeat(xv[:, None], yv.shape[-1], axis=1); return yv, xv
    if yv.ndim == 2 and xv.ndim == 2:
        C = min(yv.shape[-1], xv.shape[-1])
        return yv[:, :C], xv[:, :C]
    if yv.ndim == 1 and xv.ndim == 2:
        yv = jnp.repeat(yv[:, None], xv.shape[-1], axis=1); return yv, xv
    return yv, xv

def _center_mask(length: int, guard: int) -> jnp.ndarray:
    g = jnp.minimum(guard, length // 4)
    idx = jnp.arange(length)
    return ((idx >= g) & (idx < length - g)).astype(jnp.float32)

# ===========================
# 梯度路由：商空间相位对齐（自定义VJP）
# ===========================
@jax.custom_vjp
def align_phase_route(yhat: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    eps = 1e-8
    if yhat.ndim == 1:
        m  = _center_mask(yhat.shape[0], ROUTE_MASK_GUARD)
        zc = jnp.vdot(yhat * m, x * m)
    else:
        m  = _center_mask(yhat.shape[0], ROUTE_MASK_GUARD)[:, None]
        zc = jnp.vdot((yhat*m).reshape(-1), (x*m).reshape(-1))
    p  = zc / (jnp.abs(zc) + eps)
    return yhat * jnp.conj(p)

def _align_phase_fwd(yhat, x):
    eps = 1e-8
    if yhat.ndim == 1:
        m  = _center_mask(yhat.shape[0], ROUTE_MASK_GUARD)
        zc = jnp.vdot(yhat * m, x * m)
    else:
        m  = _center_mask(yhat.shape[0], ROUTE_MASK_GUARD)[:, None]
        zc = jnp.vdot((yhat*m).reshape(-1), (x*m).reshape(-1))
    p  = zc / (jnp.abs(zc) + eps)
    y_al = yhat * jnp.conj(p)
    return y_al, (yhat, x, p)

def _align_phase_bwd(res, g_al):
    yhat, x, p = res
    eps = 1e-8
    # 映回 yhat 坐标
    g_back = p * g_al
    # 去掉 span{yhat, i*yhat} 的规范分量（相位方向保留一部分）
    yh = yhat.reshape(-1); g = g_back.reshape(-1)
    n2 = jnp.real(jnp.vdot(yh, yh)) + eps
    u1 = yh / jnp.sqrt(n2)      # 幅度
    u2 = 1j * u1                # 相位
    c1 = jnp.vdot(u1, g)
    c2 = jnp.vdot(u2, g)
    g_vert  = u1 * c1 + (1.0 - ROUTE_KEEP_PHASE) * (u2 * c2)
    g_horiz = g - g_vert
    g_yhat = g_horiz.reshape(yhat.shape)
    g_x    = jnp.zeros_like(x)  # 不把梯度推回真值支路
    return (g_yhat, g_x)

align_phase_route.defvjp(_align_phase_fwd, _align_phase_bwd)

# ===========================
# 简单度量 + 相位一致性项
# ===========================
def si_snr_complex(tx: jnp.ndarray, rx: jnp.ndarray, eps=1e-8) -> jnp.ndarray:
    t = tx.reshape(-1); e = rx.reshape(-1)
    a = jnp.vdot(t, e) / (jnp.vdot(t, t) + eps)
    s = a * t
    err = e - s
    return -10.0 * jnp.log10(
        (jnp.real(jnp.vdot(s, s)) + eps) / (jnp.real(jnp.vdot(err, err)) + eps)
    )

def evm_norm(tx: jnp.ndarray, rx: jnp.ndarray, eps=1e-8) -> jnp.ndarray:
    return jnp.mean(jnp.abs(rx - tx)**2) / (jnp.mean(jnp.abs(tx)**2) + eps)

def phase_consistency(rx: jnp.ndarray, tx: jnp.ndarray, eps=1e-8) -> jnp.ndarray:
    """等变相位牵引：最小化相位差，幅度不参与"""
    rn = rx / (jnp.abs(rx) + eps)
    tn = tx / (jnp.abs(tx) + eps)
    # 1 - cos(Δφ) 的期望（越小越好）
    return 1.0 - jnp.real(jnp.mean(rn * jnp.conj(tn)))

# ===========================
# 模型结构（原样保留）
# ===========================
def make_base_module(steps: int = 3,
                     dtaps: int = 261,
                     ntaps: int = 41,
                     rtaps: int = 61,
                     init_fn: tuple = (core.delta, core.gauss),
                     w0=0.,
                     mode: str = 'train'):
    _assert_taps(dtaps, ntaps, rtaps)
    d_init, n_init = init_fn
    if mode == 'train':
        mimo_train = True
    elif mode == 'test':
        mimo_train = cxopt.piecewise_constant([200000], [True, False])
    else:
        raise ValueError(f'invalid mode {mode}')

    fdbp_series = layer.Serial(
        layer.FDBP(steps=steps, dtaps=dtaps, ntaps=ntaps,
                   d_init=d_init, n_init=n_init, name='fdbp1'),
        layer.BatchPowerNorm(mode=mode),
        layer.MIMOFOEAf(name='FOEAf1', w0=w0, train=mimo_train,
                        preslicer=core.conv1d_slicer(rtaps), foekwargs={}),
        layer.vmap(layer.Conv1d)(name='RConv1', taps=rtaps),
        layer.MIMOAF(train=mimo_train),
        name='fdbp_series'
    )

    serial_branch = layer.Serial(
        layer.FDBP1(steps=steps, dtaps=dtaps, ntaps=ntaps,
                    d_init=d_init, n_init=n_init),
        layer.BatchPowerNorm(mode=mode),
        layer.MIMOFOEAf(name='FOEAf', w0=w0, train=mimo_train,
                        preslicer=core.conv1d_slicer(rtaps), foekwargs={}),
        layer.vmap(layer.Conv1d)(name='RConv', taps=rtaps),
        layer.MIMOAF(train=mimo_train),
        name='serial_branch'
    )

    base = layer.Serial(
        layer.FanOut(num=2),
        layer.Parallel(fdbp_series, serial_branch),
        layer.FanInMean()
    )
    return base

def _assert_taps(dtaps, ntaps, rtaps, sps=2):
    assert dtaps % sps, f'dtaps must be odd number, got {dtaps} instead'
    assert ntaps % sps, f'ntaps must be odd number, got {ntaps} instead'
    assert rtaps % sps, f'rtaps must be odd number, got {rtaps} instead'

# ===========================
# 初始化（原逻辑）
# ===========================
def fdbp_init(a: dict, xi: float = 1.1, steps: Optional[int] = None):
    def d_init(key, shape, dtype=jnp.complex64):
        dtaps = shape[0]
        d0, _ = comm.dbp_params(
            a['samplerate'], a['distance'] / a['spans'], a['spans'],
            dtaps, a['lpdbm'] - 3, virtual_spans=steps)
        return d0[0, :, 0]
    def n_init(key, shape, dtype=jnp.float32):
        dtaps = shape[0]
        _, n0 = comm.dbp_params(
            a['samplerate'], a['distance'] / a['spans'], a['spans'],
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
    af_state   = v0.get('af_state',   freeze({}))
    aux_inputs = v0.get('aux_inputs', freeze({}))
    const      = v0.get('const',      freeze({}))
    state = {'af_state': af_state}
    return Model(mod, (params, state, aux_inputs, const, sparams), ol, name)

# ===========================
# 损失（商空间 + 相位一致性）
# ===========================
def loss_fn(module: layer.Layer,
            params: Dict,
            state : Dict,
            y     : Array,
            x     : Array,
            aux   : Dict,
            const : Dict,
            sparams: Dict,
            β_ce : float = BETA_CE,
            λ_kl : float = LAMBDA_KL):

    params_net = util.dict_merge(params, sparams)
    z_out, state_new = module.apply(
        {'params': params_net, 'aux_inputs': aux, 'const': const, **state},
        core.Signal(y))

    yhat  = z_out.val
    x_ref = x[z_out.t.start:z_out.t.stop]
    yv, xv = _match_y_x(yhat, x_ref)

    # 商空间对齐（自定义VJP，反向剔除规范分量）
    y_al = align_phase_route(yv, xv)

    # 任务损失（相位不变）+ 相位一致性（相位等变）
    evm    = evm_norm(xv, y_al)
    snr    = si_snr_complex(xv, y_al)
    bit_ce = _bit_bce_loss_16qam(y_al, xv)
    phc    = phase_consistency(yv, xv)  # 用未对齐 yv，给FDBP保留相位牵引

    # 能量约束
    kl_ib  = 0.5 * jnp.mean(jnp.square(jnp.abs(yv)))

    loss_main  = snr + 0.1 * evm + β_ce * bit_ce + ETA_PHASE * phc
    total_loss = loss_main + λ_kl * kl_ib
    return total_loss, state_new

# ===========================
# 梯度后路由 + 裁剪
# ===========================
def _global_norm(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    sq = sum([jnp.sum(jnp.square(jnp.asarray(x))) for x in leaves if x is not None])
    return jnp.sqrt(sq + 1e-12)

def _clip_by_global_norm(tree, max_norm):
    g = _global_norm(tree)
    scale = jnp.minimum(1.0, max_norm / g)
    return jax.tree_util.tree_map(lambda t: t * scale, tree)

def _scale_grads_by_rules(grads: Dict, rules: List[Tuple[Tuple[str, ...], float]]) -> Dict:
    """按参数路径名的子串匹配缩放梯度；rules: [((patterns...), factor), ...]"""
    flat = flatten_dict(grads, keep_empty_nodes=True)
    new_flat = {}
    for k, v in flat.items():
        name = "/".join(k)
        fac = 1.0
        for pats, f in rules:
            if any(p in name for p in pats):
                fac *= f
        new_flat[k] = v * fac
    return unflatten_dict(new_flat)

def _grad_rules(i: int) -> List[Tuple[Tuple[str, ...], float]]:
    """
    训练早期（<=300步）不做干预；之后：
      - 放大 FDBP/N 分支（等变应对非线性）
      - 略抑制后级 RConv/MIMOAF（相位不变统计）
      - FOE 保持中性
    具体 pattern 需与实际 param 树匹配，如有差异请据打印的 key 调整
    """
    if i <= 300:
        return []
    return [
        (("fdbp", "FDBP", "fdbp1", "/n", "nonlinear", "n_filter"), 1.35),  # 放大等变块
        (("RConv", "RConv1", "MIMOAF"), 0.80),                             # 抑制不变块
        (("FOEAf", "FOEAf1"), 1.00),                                       # 中性
    ]

# ===========================
# 单步更新（加后路由+裁剪）
# ===========================
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
                sparams: Dict):
    params = opt.params_fn(opt_state)
    (loss, module_state), grads = value_and_grad(
        loss_fn, argnums=1, has_aux=True)(
            module, params, module_state, y, x, aux, const, sparams)

    # 后路由：按模块缩放梯度
    rules = _grad_rules(i)
    if rules:
        grads = _scale_grads_by_rules(grads, rules)

    # 全局裁剪
    grads = _clip_by_global_norm(grads, CLIP_NORM)

    opt_state = opt.update_fn(i, grads, opt_state)
    return loss, opt_state, module_state

# ===========================
# 批生成 / 训练 / 测试（与原一致，细节微调）
# ===========================
def get_train_batch(ds, batchsize: int, overlaps: int, sps: int = 2):
    flen  = batchsize + overlaps
    fstep = batchsize
    ds_y  = op.frame_gen(ds.y, flen * sps, fstep * sps)
    ds_x  = op.frame_gen(ds.x, flen, fstep)
    n_batches = op.frame_shape(ds.x.shape, flen, fstep)[0]
    return n_batches, zip(ds_y, ds_x)

def train(model: Model,
          data,
          batch_size: int = 500,
          n_iter = None,
          opt: optim.Optimizer = optim.adam(
              optim.piecewise_constant([500, 1000], [1e-4, 5e-5, 1e-5])
          )):
    params, module_state, aux, const, sparams = model.initvar
    opt_state = opt.init_fn(params)

    n_batch, batch_gen = get_train_batch(data, batch_size, model.overlaps)
    n_iter = n_batch if n_iter is None else min(n_iter, n_batch)

    for i, (y, x) in enumerate(batch_gen):
        if i >= n_iter: break
        if TF_ENABLE:
            aux = core.dict_replace(aux, {'truth': x})
        loss, opt_state, module_state = update_step(
            model.module, opt, i, opt_state, module_state, y, x, aux, const, sparams
        )
        yield loss, opt.params_fn(opt_state), module_state

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

    yhat = z.val
    xref = data.x[z.t.start:z.t.stop]
    yv, xv = _match_y_x(yhat, xref)

    # 评测对齐（相位 + 幅度）
    zc  = jnp.vdot(yv.reshape(-1), xv.reshape(-1))
    p   = zc / (jnp.abs(zc) + 1e-12)
    yph = yv * jnp.conj(p)
    num = jnp.real(jnp.vdot(yph.reshape(-1), xv.reshape(-1)))
    den = jnp.real(jnp.vdot(yph.reshape(-1), yph.reshape(-1))) + 1e-12
    y_eval = (num/den) * yph

    metric = metric_fn(y_eval, xv, scale=np.sqrt(10), eval_range=eval_range)
    return metric, z

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

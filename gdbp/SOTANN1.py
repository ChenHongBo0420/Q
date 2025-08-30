# =========================================
# QAL 版：商空间对齐 + 旁观自适应（DROP-IN）
# =========================================
from jax import numpy as jnp, random, jit, value_and_grad, lax
import jax.nn as jnn
import flax
from flax import linen as flnn
from functools import partial
import numpy as np

from commplax import util, comm, cxopt, op, optim
from commplax.module import core, layer

Array = any
Dict  = (dict, flax.core.FrozenDict)

# ---------------------------
# 0) 安全的 No-Op 模块（若无 layer.Identity）
# ---------------------------
try:
    _ = layer.Identity
    def _id_block(name):
        return layer.Identity(name=name)
except Exception:
    def _NoOp(scope, signal, **kwargs):
        return signal
    def _id_block(name):
        return _NoOp

# ---------------------------
# 1) 基础模块：训练期 use_qal=True → 旁路 FOE/MIMOAF
# ---------------------------
def make_base_module(steps: int = 3,
                     dtaps: int = 261,
                     ntaps: int = 41,
                     rtaps: int = 61,
                     init_fn: tuple = (core.delta, core.gauss),
                     w0=0.,
                     mode: str = 'train',
                     use_qal: bool = False):       # ★ 新增：训练期旁路开关
    _assert_taps(dtaps, ntaps, rtaps)
    d_init, n_init = init_fn

    if mode == 'train' and use_qal:
        # 训练时：彻底旁路 FOE/MIMOAF，避免“吃梯度”
        foe_block  = _id_block('FOE_off')
        mimo_block = _id_block('MIMO_off')
    else:
        # 测试（或非 QAL 训练）：让 FOE/MIMOAF 正常工作
        mimo_train = (mode == 'train')
        foe_block  = layer.MIMOFOEAf(name='FOEAf',
                                     w0=w0, train=mimo_train,
                                     preslicer=core.conv1d_slicer(rtaps),
                                     foekwargs={})
        mimo_block = layer.MIMOAF(train=mimo_train)

    fdbp_series = layer.Serial(
        layer.FDBP(steps=steps, dtaps=dtaps, ntaps=ntaps,
                   d_init=d_init, n_init=n_init, name='fdbp1'),
        layer.BatchPowerNorm(mode=mode),
        foe_block,                                   # ← 由开关控制
        layer.vmap(layer.Conv1d)(name='RConv1', taps=rtaps),
        mimo_block,
        name='fdbp_series'
    )

    serial_branch = layer.Serial(
        layer.FDBP1(steps=steps, dtaps=dtaps, ntaps=ntaps,
                    d_init=d_init, n_init=n_init),
        layer.BatchPowerNorm(mode=mode),
        foe_block,                                   # ← 同步
        layer.vmap(layer.Conv1d)(name='RConv', taps=rtaps),
        mimo_block,
        name='serial_branch'
    )

    base = layer.Serial(
        layer.FanOut(num=2),
        layer.Parallel(fdbp_series, serial_branch),
        layer.FanInMean()
    )
    return base

def _assert_taps(dtaps, ntaps, rtaps, sps=2):
    assert dtaps % sps, f'dtaps must be odd, got {dtaps}'
    assert ntaps % sps, f'ntaps must be odd, got {ntaps}'
    assert rtaps % sps, f'rtaps must be odd, got {rtaps}'

def fdbp_init(a: dict, xi: float = 1.1, steps: int | None = None):
    def d_init(key, shape, dtype=jnp.complex64):
        dtaps = shape[0]
        d0, _ = comm.dbp_params(a['samplerate'], a['distance']/a['spans'], a['spans'],
                                dtaps, a['lpdbm']-3, virtual_spans=steps)
        return d0[0, :, 0]
    def n_init(key, shape, dtype=jnp.float32):
        dtaps = shape[0]
        _, n0 = comm.dbp_params(a['samplerate'], a['distance']/a['spans'], a['spans'],
                                dtaps, a['lpdbm']-3, virtual_spans=steps)
        return xi * n0[0, 0, 0] * core.gauss(key, shape, dtype)
    return d_init, n_init

# ---------------------------
# 2) 商空间对齐的任务一致损失（相位 only + stop-grad）
# ---------------------------
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
    logits = -jnp.square(jnp.abs(pred_sym[..., None] - CONST_16QAM))     # [N,16]
    logp   = logits - jnn.logsumexp(logits, axis=-1, keepdims=True)
    probs  = jnp.exp(logp)                                               # [N,16]
    p1     = (probs @ BIT_MAP)                                           # [N,4]
    p0     = 1.0 - p1
    idx    = jnp.argmin(jnp.square(jnp.abs(true_sym[..., None] - CONST_16QAM)), axis=-1)
    bits   = BIT_MAP[idx]                                                # [N,4]
    bce    = -(bits*jnp.log(p1+1e-12) + (1.-bits)*jnp.log(p0+1e-12))
    return (bce * BIT_WEIGHTS).mean()

def loss_fn(module: layer.Layer,
            params: Dict,
            state : Dict,
            y     : Array,
            x     : Array,
            aux   : Dict,
            const : Dict,
            sparams: Dict,
            β_ce : float = 0.5,
            λ_kl : float = 1e-4):
    """
    QAL: 相位商空间对齐 + stop-grad（不做幅度对齐）
      - 在 y_al 上计算 EVM/SI-SNR/Bit-BCE
      - KL 仍作用在未对齐 yhat 上（能量正则）
    """
    params_net = util.dict_merge(params, sparams)
    z_out, state_new = module.apply(
        {'params': params_net, 'aux_inputs': aux, 'const': const, **state},
        core.Signal(y)
    )
    x_ref = x[z_out.t.start:z_out.t.stop]    # 参考符号（裁剪后）
    yhat  = z_out.val                        # 预测复符号

    # —— 相位对齐（闭式）+ stop-grad ——
    zc   = jnp.vdot(yhat.reshape(-1), x_ref.reshape(-1)) # <ŷ, x>
    p    = zc / (jnp.abs(zc) + 1e-8)                     # e^{jφ*}
    psg  = lax.stop_gradient(jnp.conj(p))                # ★ 不反传 φ*
    y_al = yhat * psg                                    # 去相位后的输出

    # —— 任务一致计分（在 y_al 上）——
    # 1) EVM（Q²代理）
    evm = (jnp.mean(jnp.abs(y_al - x_ref)**2) /
           (jnp.mean(jnp.abs(x_ref)**2) + 1e-8))
    # 2) SI-SNR（复数版）
    t = x_ref.reshape(-1); e = y_al.reshape(-1)
    a = jnp.vdot(t, e) / (jnp.vdot(t, t) + 1e-8)
    s = a * t; err = e - s
    snr = -10.0 * jnp.log10(
        (jnp.real(jnp.vdot(s, s)) + 1e-8) /
        (jnp.real(jnp.vdot(err, err)) + 1e-8)
    )
    # 3) Bit-BCE
    bit_bce = _bit_bce_loss_16qam(y_al, x_ref)

    # —— IB-KL 正则（作用在未对齐的 yhat）——
    kl_ib = 0.5 * jnp.mean(jnp.square(jnp.abs(yhat)))

    loss_main  = snr + 0.1 * evm + β_ce * bit_bce
    total_loss = loss_main + λ_kl * kl_ib
    return total_loss, state_new

# ---------------------------
# 3) 训练/测试：不再喂 truth 给自适应
# ---------------------------
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
        loss_fn, argnums=1, has_aux=True)(module, params, module_state, y, x,
                                          aux, const, sparams)
    opt_state = opt.update_fn(i, grads, opt_state)
    return loss, opt_state, module_state

def get_train_batch(ds, batchsize: int, overlaps: int, sps: int = 2):
    flen = batchsize + overlaps
    fstep = batchsize
    ds_y = op.frame_gen(ds.y, flen * sps, fstep * sps)
    ds_x = op.frame_gen(ds.x, flen, fstep)
    n_batches = op.frame_shape(ds.x.shape, flen, fstep)[0]
    return n_batches, zip(ds_y, ds_x)

def train(model,
          data,
          batch_size: int = 500,
          n_iter = None,
          opt: optim.Optimizer = optim.adam(optim.piecewise_constant([500, 1000], [1e-4, 1e-5, 1e-6]))):
    """QAL 训练：不把 truth 喂入 aux，自适应已被旁路"""
    params, module_state, aux, const, sparams = model.initvar
    opt_state = opt.init_fn(params)
    n_batch, batch_gen = get_train_batch(data, batch_size, model.overlaps)
    n_iter = n_batch if n_iter is None else min(n_iter, n_batch)

    for i, (y, x) in enumerate(batch_gen):
        if i >= n_iter: break
        # ★ 不要 teacher-forcing（不要喂 truth）
        # aux = core.dict_replace(aux, {'truth': x})
        loss, opt_state, module_state = update_step(model.module, opt, i, opt_state,
                                                   module_state, y, x, aux,
                                                   const, sparams)
        yield loss, opt.params_fn(opt_state), module_state

def test(model,
         params,
         data,
         eval_range: tuple=(300000, -20000),
         metric_fn=comm.qamqot):
    """测试：同样不喂 truth，让 FOE/MIMOAF 以 DD/无参模式运行"""
    state, aux, const, sparams = model.initvar[1:]
    # aux = core.dict_replace(aux, {'truth': data.x})  # ★ 不要喂
    if params is None:
        params = model.initvar[0]

    z, _ = jit(model.module.apply, backend='cpu')({
        'params': util.dict_merge(params, sparams),
        'aux_inputs': aux,
        'const': const,
        **state
    }, core.Signal(data.y))
    metric = metric_fn(z.val,
                       data.x[z.t.start:z.t.stop],
                       scale=np.sqrt(10),
                       eval_range=eval_range)
    return metric, z

# ---------------------------
# 4) 初始化：训练用 QAL，测试正常
# ---------------------------
def model_init(data, base_conf: dict, sparams_flatkeys: list,
               n_symbols: int = 4000, sps: int = 2, name='Model'):
    mod = make_base_module(**base_conf, w0=data.w0)
    y0 = data.y[:n_symbols * sps]
    rng0 = random.PRNGKey(0)
    z0, v0 = mod.init(rng0, core.Signal(y0))
    ol = z0.t.start - z0.t.stop
    sparams, params = util.dict_split(v0['params'], sparams_flatkeys)
    state = v0['af_state']; aux = v0['aux_inputs']; const = v0['const']
    from collections import namedtuple
    Model = namedtuple('Model', 'module initvar overlaps name')
    return Model(mod, (params, state, aux, const, sparams), ol, name)

def init_models_qal(data_train, data_test,
                    steps=3, dtaps=271, ntaps=71, rtaps=321, xi=1.1):
    fi_train = partial(fdbp_init, data_train.a, steps=steps)
    fi_test  = partial(fdbp_init, data_test.a,  steps=steps)

    base_conf_train = dict(mode='train', steps=steps, dtaps=dtaps, ntaps=ntaps,
                           rtaps=rtaps, init_fn=fi_train(xi=xi), use_qal=True)   # ★
    base_conf_test  = dict(mode='test',  steps=steps, dtaps=dtaps, ntaps=ntaps,
                           rtaps=rtaps, init_fn=fi_test(xi=xi))                  # ★

    cdc_tr  = model_init(data_train, {**base_conf_train, 'ntaps': 1, 'init_fn': fi_train(xi=0.0)},
                         sparams_flatkeys=[('fdbp_0',)], name='CDC-train')
    gdbp_tr = model_init(data_train, base_conf_train, sparams_flatkeys=[],
                         name='GDBP-train')

    cdc_te  = model_init(data_test, {**base_conf_test, 'ntaps': 1, 'init_fn': fi_test(xi=0.0)},
                         sparams_flatkeys=[('fdbp_0',)], name='CDC-test')
    gdbp_te = model_init(data_test, base_conf_test, sparams_flatkeys=[],
                         name='GDBP-test')
    return (cdc_tr, gdbp_tr), (cdc_te, gdbp_te)

# =========================================
# Minimal change: Orbit-averaged bit-BCE (angles: 0, pi/2)
# 其他结构/训练流程不动；去除冗余代码与未用依赖
# =========================================

from jax import numpy as jnp, random, jit, value_and_grad
import jax
from jax import lax
import flax
from commplax import util, comm1, cxopt, op, optim
from commplax.module import core, layer
import numpy as np
from functools import partial
from collections import namedtuple
from tqdm.auto import tqdm
from typing import Any, Optional, Union, Tuple
from . import data as gdat

Model = namedtuple('Model', 'module initvar overlaps name')
Array = Any
Dict = Union[dict, flax.core.FrozenDict]

# -----------------------------------------------------------------------------
# Base module (原样保留)
# -----------------------------------------------------------------------------
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
        raise ValueError('invalid mode %s' % mode)

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

# -----------------------------------------------------------------------------
# FDBP init (原样)
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Model init (原样)
# -----------------------------------------------------------------------------
def model_init(data: gdat.Input,
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
    state = v0['af_state']
    aux = v0['aux_inputs']
    const = v0['const']
    return Model(mod, (params, state, aux, const, sparams), ol, name)

# -----------------------------------------------------------------------------
# 16QAM map & bit weights（用于 CE/BCE）
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Orbit-averaged bit-BCE（唯一的功能性改动）
# -----------------------------------------------------------------------------
def _orbit_bce_16qam(y: Array, x: Array, tau: float, angles=(0.0, jnp.pi/2)):
    """对同一(y,x)在两个相位(0, π/2)下计算 bit-BCE 并平均，提升相位稳健性。"""
    def one(theta):
        u   = jnp.exp(1j * theta)
        yr  = y * u
        xr  = x * u
        logits = -jnp.square(jnp.abs(yr[..., None] - CONST_16QAM)) / (tau * tau)  # [...,16]
        logp   = logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True)
        probs  = jnp.exp(logp)
        p1     = (probs @ BIT_MAP)
        p0     = 1.0 - p1
        idx    = jnp.argmin(jnp.square(jnp.abs(xr[..., None] - CONST_16QAM)), axis=-1)
        bits   = BIT_MAP[idx]
        bce    = -(bits * jnp.log(p1 + 1e-12) + (1. - bits) * jnp.log(p0 + 1e-12))
        return (bce * BIT_WEIGHTS).mean()
    losses = [one(th) for th in angles]   # angles 是常量 tuple，JIT 安全
    return sum(losses) / len(losses)

# -----------------------------------------------------------------------------
# Metrics（原配方）
# -----------------------------------------------------------------------------
def _evm_norm(tx: jnp.ndarray, rx: jnp.ndarray, eps=1e-8) -> jnp.ndarray:
    return jnp.mean(jnp.abs(rx - tx)**2) / (jnp.mean(jnp.abs(tx)**2) + eps)

def _si_snr_complex(tx: jnp.ndarray, rx: jnp.ndarray, eps=1e-8) -> jnp.ndarray:
    t = tx.reshape(-1); e = rx.reshape(-1)
    a = jnp.vdot(t, e) / (jnp.vdot(t, t) + eps)
    s = a * t
    err = e - s
    return -10.0 * jnp.log10(
        (jnp.real(jnp.vdot(s, s)) + eps) / (jnp.real(jnp.vdot(err, err)) + eps)
    )

# -----------------------------------------------------------------------------
# Loss（只把原来的 bit-BCE 换成 orbit-averaged 版本；其余不动）
# -----------------------------------------------------------------------------
def loss_fn(module, params, state, y, x, aux, const, sparams,
            β_ce: float = 0.5, λ_kl: float = 1e-4,
            *,
            ETA_PHASE: float = 0.5,   # 只去除 η·φ*（温和）
            TAU_BCE:  float = 1.5,    # bit-BCE 温度
            TAIL_WIN: int   = 16384   # 尾窗估计全局相位（静态切片）
            ):
    # 前向
    params_net = util.dict_merge(params, sparams)
    z_out, state_new = module.apply(
        {'params': params_net, 'aux_inputs': aux, 'const': const, **state},
        core.Signal(y)
    )

    # 裁剪后的参考与预测
    x_ref = x[z_out.t.start:z_out.t.stop]
    yhat  = z_out.val

    # 用固定尾窗估计全局相位（估计不回传）
    T  = int(yhat.shape[0])
    W  = min(TAIL_WIN, T)
    ys = yhat[-W:].reshape(-1)
    xs = x_ref[-W:].reshape(-1)
    zc = jnp.vdot(ys, xs)
    p  = zc / (jnp.abs(zc) + 1e-8)
    p  = lax.stop_gradient(p)
    phi = jnp.angle(p)

    # 主损里仅去除 η·φ*（不做幅度对齐）
    phase_factor = jnp.exp(-1j * ETA_PHASE * phi)
    y_al = yhat * phase_factor

    # EVM & SI-SNR（原配方）
    evm = _evm_norm(x_ref, y_al)
    snr = _si_snr_complex(x_ref, y_al)

    # ★ 改动：bit-BCE 使用 U(1) 轨道平均
    bit_bce = _orbit_bce_16qam(y_al, x_ref, tau=TAU_BCE, angles=(0.0, jnp.pi/2))

    # 信息瓶颈（未对齐输出）
    kl_ib = 0.5 * jnp.mean(jnp.square(jnp.abs(yhat)))

    loss_main  = snr + 0.1 * evm + β_ce * bit_bce
    total_loss = loss_main + λ_kl * kl_ib
    return total_loss, state_new

# -----------------------------------------------------------------------------
# Train/update/test（原样）
# -----------------------------------------------------------------------------
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
    opt_state = opt.update_fn(i, grads, opt_state)
    return loss, opt_state, module_state

def get_train_batch(ds: gdat.Input,
                    batchsize: int,
                    overlaps: int,
                    sps: int = 2):
    flen  = batchsize + overlaps
    fstep = batchsize
    ds_y  = op.frame_gen(ds.y, flen * sps, fstep * sps)
    ds_x  = op.frame_gen(ds.x, flen, fstep)
    n_batches = op.frame_shape(ds.x.shape, flen, fstep)[0]
    return n_batches, zip(ds_y, ds_x)

def train(model: Model,
          data: gdat.Input,
          batch_size: int = 500,
          n_iter = None,
          opt: optim.Optimizer = optim.adam(
              optim.piecewise_constant([500, 1000], [1e-4, 1e-5, 1e-6]))):
    params, module_state, aux, const, sparams = model.initvar
    opt_state = opt.init_fn(params)

    n_batch, batch_gen = get_train_batch(data, batch_size, model.overlaps)
    n_iter = n_batch if n_iter is None else min(n_iter, n_batch)

    for i, (y, x) in tqdm(enumerate(batch_gen), total=n_iter, desc='training', leave=False):
        if i >= n_iter: break
        aux = core.dict_replace(aux, {'truth': x})
        loss, opt_state, module_state = update_step(
            module=model.module, opt=opt, i=i,
            opt_state=opt_state, module_state=module_state,
            y=y, x=x, aux=aux, const=const, sparams=sparams
        )
        yield loss, opt.params_fn(opt_state), module_state

def test(model: Model,
         params: Dict,
         data: gdat.Input,
         eval_range: tuple = (300000, -20000),
         L: int = None,                 # 每复维星座大小（16/64/256...）；不填则按旧口径假定16QAM缩放
         d4_dims: int = 2,              # DP=2（4D）
         pilot_frac: float = 0.0,       # 导频占比（算 AIR 用）
         use_elliptical_llr: bool = True):
    state, aux, const, sparams = model.initvar[1:]
    aux = core.dict_replace(aux, {'truth': data.x})
    if params is None:
        params = model.initvar[0]

    z, _ = jit(model.module.apply, backend='cpu')({
        'params': util.dict_merge(params, sparams),
        'aux_inputs': aux, 'const': const, **state
    }, core.Signal(data.y))

    # 参考序列与缩放
    xref = data.x[z.t.start:z.t.stop]
    # 若给了 L，则用 canonical 缩放；否则沿用你们原先的 sqrt(10)（相当于16QAM口径）
    scale = comm1.qamscale(L) if L is not None else np.sqrt(10)

    # 直接用扩展指标（含 BER/Q/EVM/SNR、GMI/NGMI/AIR、C4D与Gap）
    metric = comm.qamqot_ext(
        z.val, xref,
        L=L,
        d4_dims=d4_dims,
        eval_range=eval_range,
        scale=scale,
        pilot_frac=pilot_frac,
        use_elliptical_llr=use_elliptical_llr
    )
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


# [np.float64(8.263075435507282), np.float64(8.809552824900607)]
# [np.float64(0.004810945210357634), np.float64(0.002914403700453229)]

# Copyright 2021 The Commplax Authors.
# Modified: add split-loss + gradient routing (equivariant main loss, invariant aux loss)

from jax import numpy as jnp, random, jit, value_and_grad, lax
import flax
from flax import linen as nn
from flax.core import freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
from commplax import util, comm, cxopt, op, optim
from commplax.module import core, layer
import numpy as np
from functools import partial
from collections import namedtuple
from tqdm.auto import tqdm
from typing import Any, Optional, Union
import jax
import matplotlib.pyplot as plt

# ======= Aliases =======
Model = namedtuple('Model', 'module initvar overlaps name')
Array = Any
Dict = Union[dict, flax.core.FrozenDict]

# ========== Base module (unchanged) ==========
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
        layer.FDBP(steps=steps,
                   dtaps=dtaps,
                   ntaps=ntaps,
                   d_init=d_init,
                   n_init=n_init),
        layer.BatchPowerNorm(mode=mode),
        layer.MIMOFOEAf(name='FOEAf',
                        w0=w0,
                        train=mimo_train,
                        preslicer=core.conv1d_slicer(rtaps),
                        foekwargs={}),
        layer.vmap(layer.Conv1d)(name='RConv', taps=rtaps),  # vectorize column-wise Conv1D
        layer.MIMOAF(train=mimo_train))  # adaptive MIMO layer
        
    return base
  

def _assert_taps(dtaps, ntaps, rtaps, sps=2):
    assert dtaps % sps, f'dtaps must be odd number, got {dtaps} instead'
    assert ntaps % sps, f'ntaps must be odd number, got {ntaps} instead'
    assert rtaps % sps, f'rtaps must be odd number, got {rtaps} instead'


def fdbp_init(a: dict,
              xi: float = 1.1,
              steps: Optional[int] = None):
    def d_init(key, shape, dtype=jnp.complex64):
        dtaps = shape[0]
        d0, _ = comm.dbp_params(
            a['samplerate'],
            a['distance'] / a['spans'],
            a['spans'],
            dtaps,
            a['lpdbm'] - 3,
            virtual_spans=steps)
        return d0[0, :, 0]

    def n_init(key, shape, dtype=jnp.float32):
        dtaps = shape[0]
        _, n0 = comm.dbp_params(
            a['samplerate'],
            a['distance'] / a['spans'],
            a['spans'],
            dtaps,
            a['lpdbm'] - 3,
            virtual_spans=steps)

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
    state = v0['af_state']
    aux = v0['aux_inputs']
    const = v0['const']
    return Model(mod, (params, state, aux, const, sparams), ol, name)

# ======= helpers (kept) =======
def l2_normalize(x, axis=None, epsilon=1e-12):
    square_sum = jnp.sum(jnp.square(x), axis=axis, keepdims=True)
    x_inv_norm = jnp.sqrt(jnp.maximum(square_sum, epsilon))
    return x / x_inv_norm

def _estimate_kmean(y, x, blk=16384):
    kap=[]
    for i in range(0, len(x)-blk, blk):
        kap.append(np.abs(np.vdot(y[i:i+blk], x[i:i+blk]) /
                          np.vdot(x[i:i+blk], x[i:i+blk])))
    return float(np.mean(kap))

def _rms(sig, eps=1e-12):
    return jnp.sqrt(jnp.mean(jnp.abs(sig)**2) + eps)

def _proj_mse(z, s):
    alpha = jnp.vdot(z, s) / jnp.vdot(s, s)
    return jnp.mean(jnp.abs(z - alpha*s)**2)

def energy(x):
    return jnp.sum(jnp.square(x))

def si_snr(target, estimate, eps=1e-8):
    target_energy = energy(target)
    dot_product = jnp.sum(target * estimate)
    s_target = dot_product / (target_energy + eps) * target
    e_noise = estimate - s_target
    target_energy = energy(s_target)
    noise_energy = energy(e_noise)
    si_snr_value = 10 * jnp.log10((target_energy + eps) / (noise_energy + eps))
    return -si_snr_value 

def evm_ring(tx, rx, eps=1e-8,
             thr_in=0.60, thr_mid=1.10,
             w_in=1.0, w_mid=1.5, w_out=2.0):
    r    = jnp.abs(tx)
    err2 = jnp.abs(rx - tx)**2
    sig2 = jnp.abs(tx)**2
    m_in  = (r < thr_in).astype(jnp.float32)
    m_mid = ((r >= thr_in) & (r < thr_mid)).astype(jnp.float32)
    m_out = (r >= thr_mid).astype(jnp.float32)
    def _evm(mask):
        num = jnp.sum(err2 * mask)
        den = jnp.sum(sig2 * mask)
        den = jnp.where(den < 1e-8, 1e-8, den)
        return num / den
    return (w_in  * _evm(m_in) +
            w_mid * _evm(m_mid) +
            w_out * _evm(m_out))

def si_snr_flat_amp_pair(tx, rx, eps=1e-8):
    s  = jnp.reshape(jnp.abs(tx), (-1,))
    x  = jnp.reshape(jnp.abs(rx), (-1,))
    alpha = jnp.vdot(s, x).real / (jnp.vdot(s, s).real + eps)
    e  = x - alpha * s
    snr_db = 10. * jnp.log10( (jnp.vdot(alpha*s, alpha*s).real + eps) /
                              (jnp.vdot(e, e).real + eps) )
    return -snr_db                   

def align_phase_scale(yhat: jnp.ndarray, y: jnp.ndarray, eps: float = 1e-8):
    z = jnp.vdot(yhat, y)
    p = z / (jnp.abs(z) + eps)
    yhp = yhat * jnp.conj(p)
    alpha = jnp.real(jnp.vdot(yhp, y)) / (jnp.real(jnp.vdot(yhp, yhp)) + eps)
    return alpha * yhp, alpha, jnp.angle(p)

# ====== 16QAM constellation & bit maps (for aux loss) ======
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
BIT_MAP = jnp.array(_bits, dtype=jnp.float32)          # [16,4]
BIT_WEIGHTS = jnp.array([1.2, 1.0, 1.0, 0.8], dtype=jnp.float32)

# ===========================================================
#      NEW: parameter mask (route aux-grad only to back-end)
# ===========================================================
_BACK_KEYS = ('FOEAf', 'RConv', 'MIMOAF', 'CDC', 'Demap', 'demap')  # edit to your names

def make_back_mask(params: Dict) -> Dict:
    """Return a mask tree (1 for back-end params, 0 otherwise) with same structure as params."""
    flat = flatten_dict(unfreeze(params))  # keys = tuple path
    mflat = {}
    for path, leaf in flat.items():
        keystr = '/'.join([str(p) for p in path])
        hit = any(tag in keystr for tag in _BACK_KEYS)
        mflat[path] = jnp.ones_like(leaf) if hit else jnp.zeros_like(leaf)
    return freeze(unflatten_dict(mflat))

# ===========================================================
#            NEW: split loss (main equivariant / aux invariant)
# ===========================================================
def loss_main_fn(module, params, state, y, x, aux, const, sparams, λ_kl: float = 1e-4):
    """
    主损（等变）：原复域 MSE + 轻量能量正则（给前端/LDBP 提供相位敏感梯度）
    """
    params_net = util.dict_merge(params, sparams)
    z_out, state_new = module.apply(
        {'params': params_net, 'aux_inputs': aux, 'const': const, **state},
        core.Signal(y)
    )
    x_ref = x[z_out.t.start:z_out.t.stop]
    yhat  = z_out.val
    mse = jnp.mean(jnp.abs(yhat - x_ref)**2)
    reg = 0.5 * jnp.mean(jnp.square(jnp.abs(yhat)))
    loss = mse + λ_kl * reg
    return loss, state_new

def loss_aux_fn(module, params, state, y, x, aux, const, sparams, tau: float = 1.5):
    """
    副损（不变）：先估全局相位 φ* 对齐，再在商空间上做 bit-BCE（只给末端/CDC）
    相位估计 stop-grad，禁止回传到前端
    """
    params_net = util.dict_merge(params, sparams)
    z_out, state_new = module.apply(
        {'params': params_net, 'aux_inputs': aux, 'const': const, **state},
        core.Signal(y)
    )
    x_ref = x[z_out.t.start:z_out.t.stop]
    yhat  = z_out.val

    # 估计全局相位（尾窗），并 stop-grad
    TAIL_WIN = 16384
    T  = int(yhat.shape[0]); W = min(TAIL_WIN, T)
    ys = yhat[-W:].reshape(-1)
    xs = x_ref[-W:].reshape(-1)
    zc = jnp.vdot(ys, xs)
    p  = lax.stop_gradient(zc / (jnp.abs(zc) + 1e-8))  # 单位相位
    y_al = yhat * jnp.conj(p)

    # bit-BCE（16QAM）
    logits = -jnp.square(jnp.abs(y_al[..., None] - CONST_16QAM)) / (tau * tau)  # [...,16]
    logp   = logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True)
    probs  = jnp.exp(logp)
    p1     = (probs @ BIT_MAP); p0 = 1.0 - p1

    idx    = jnp.argmin(jnp.square(jnp.abs(x_ref[..., None] - CONST_16QAM)), axis=-1)
    bits   = BIT_MAP[idx]
    bce    = -(bits * jnp.log(p1 + 1e-12) + (1. - bits) * jnp.log(p0 + 1e-12))
    loss   = (bce * BIT_WEIGHTS).mean()
    return loss, state_new

# ===========================================================
#                UPDATED: update_step (gradient routing)
# ===========================================================
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
                mask_back: Dict):
    """
    单步更新（梯度路由版）：
      - g_main：主损梯度（等变），更新所有可训练参数
      - g_aux ：副损梯度（不变），只路由到 '末端' 参数（mask_back==1）
    """
    params = opt.params_fn(opt_state)

    # 主损梯度
    (Lm, state_main), g_main = value_and_grad(
        loss_main_fn, argnums=1, has_aux=True)(
        module, params, module_state, y, x, aux, const, sparams
    )
    # 副损梯度（状态不采用，只取梯度）
    (La, _), g_aux = value_and_grad(
        loss_aux_fn, argnums=1, has_aux=True)(
        module, params, module_state, y, x, aux, const, sparams
    )

    # 路由：副损梯度只加到末端参数
    g_routed = jax.tree_map(lambda gm, ga, mb: gm + alpha_aux * ga * mb,
                            g_main, g_aux, mask_back)

    opt_state = opt.update_fn(i, g_routed, opt_state)
    return (Lm, La, Lm + alpha_aux * La), opt_state, state_main

# ===========================================================
#                     Data batching (unchanged)
# ===========================================================
def get_train_batch(ds,
                    batchsize: int,
                    overlaps: int,
                    sps: int = 2):
    flen = batchsize + overlaps
    fstep = batchsize
    ds_y = op.frame_gen(ds.y, flen * sps, fstep * sps)
    ds_x = op.frame_gen(ds.x, flen, fstep)
    n_batches = op.frame_shape(ds.x.shape, flen, fstep)[0]
    return n_batches, zip(ds_y, ds_x)

# ===========================================================
#                 UPDATED: train (add alpha_aux & mask)
# ===========================================================
def train(model: Model,
          data,
          batch_size: int = 500,
          n_iter = None,
          alpha_aux: float = 0.02,   # ☆ 副损权重（0.01~0.05）
          opt: optim.Optimizer = optim.adam(optim.piecewise_constant([500, 1000], [1e-4, 1e-5, 1e-6]))):
    params, module_state, aux, const, sparams = model.initvar
    opt_state = opt.init_fn(params)

    # ☆ 只需构造一次末端掩膜（与参数树同形状）
    mask_back = make_back_mask(params)

    n_batch, batch_gen = get_train_batch(data, batch_size, model.overlaps)
    n_iter = n_batch if n_iter is None else min(n_iter, n_batch)

    for i, (y, x) in tqdm(enumerate(batch_gen), total=n_iter, desc='training', leave=False):
        if i >= n_iter: break
        aux = core.dict_replace(aux, {'truth': x})

        (Lm, La, Ltot), opt_state, module_state = update_step(
            model.module, opt, i, opt_state, module_state,
            y, x, aux, const, sparams,
            alpha_aux, mask_back
        )
        yield float(Ltot), opt.params_fn(opt_state), module_state

# ===========================================================
#                           test (unchanged)
# ===========================================================
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

    metric = metric_fn(
        z.val, data.x[z.t.start:z.t.stop],
        scale=np.sqrt(10), eval_range=eval_range
    )
    return metric, z

# ===========================================================
#                equalize_dataset (unchanged)
# ===========================================================
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


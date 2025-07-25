# CE/BCE loss weighted bit-BCE
from jax import numpy as jnp, random, jit, value_and_grad, nn
import flax
from commplax import util, comm, cxopt, op, optim
from commplax.module import core, layer
import numpy as np
from functools import partial
from collections import namedtuple
from tqdm.auto import tqdm
from typing import Any, Optional, Union, Tuple
from . import data as gdat
import jax
from scipy import signal
from flax import linen as nn
from sklearn.neighbors import KernelDensity
from jax.scipy.stats import norm
from jax import jit, lax
from typing import Tuple
import matplotlib.pyplot as plt
from flax.core import unfreeze, freeze
Signal = core.Signal
Model = namedtuple('Model', 'module initvar overlaps name')
Array = Any
Dict = Union[dict, flax.core.FrozenDict]


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
        # configure mimo to its training mode
        mimo_train = True
    elif mode == 'test':
        # mimo operates at training mode for the first 200000 symbols,
        # then switches to tracking mode afterwards
        mimo_train = cxopt.piecewise_constant([200000], [True, False])
    else:
        raise ValueError('invalid mode %s' % mode)
        
    base = layer.Serial(
        layer.FDBP(steps=steps,
                   dtaps=dtaps,
                   ntaps=ntaps,
                   d_init=d_init,
                   n_init=n_init),
        layer.BatchPowerNorm(mode=mode),)
        # layer.MIMOFOEAf(name='FOEAf',
        #                 w0=w0,
        #                 train=mimo_train,
        #                 preslicer=core.conv1d_slicer(rtaps),
        #                 foekwargs={}),
        # layer.vmap(layer.Conv1d)(name='RConv', taps=rtaps),  # vectorize column-wise Conv1D
        # layer.MIMOAF(train=mimo_train))  # adaptive MIMO layer
        
    return base

def _assert_taps(dtaps, ntaps, rtaps, sps=2):
    ''' we force odd taps to ease coding '''
    assert dtaps % sps, f'dtaps must be odd number, got {dtaps} instead'
    assert ntaps % sps, f'ntaps must be odd number, got {ntaps} instead'
    assert rtaps % sps, f'rtaps must be odd number, got {rtaps} instead'


def fdbp_init(a: dict,
              xi: float = 1.1,
              steps: Optional[int] = None):
    '''
        initializer for the base module

        Args:
            xi: NLC scaling factor
            steps: GDBP steps, used to calculate the theoretical profiles of D- and N-filters

        Returns:
            a pair of functions to initialize D- and N-filters
    '''

    def d_init(key, shape, dtype=jnp.complex64):
        dtaps = shape[0]
        d0, _ = comm.dbp_params(
            a['samplerate'],
            a['distance'] / a['spans'],
            a['spans'],
            dtaps,
            a['lpdbm'] - 3,  # rescale as input power which has been norm to 2 in dataloader
            virtual_spans=steps)
        return d0[0, :, 0]

    def n_init(key, shape, dtype=jnp.float32):
        dtaps = shape[0]
        _, n0 = comm.dbp_params(
            a['samplerate'],
            a['distance'] / a['spans'],
            a['spans'],
            dtaps,
            a['lpdbm'] - 3,  # rescale
            virtual_spans=steps)

        return xi * n0[0, 0, 0] * core.gauss(key, shape, dtype)

    return d_init, n_init


# def model_init(data: gdat.Input,
#                base_conf: dict,
#                sparams_flatkeys: list,
#                n_symbols: int = 4000,
#                sps : int = 2,
#                name='Model'):
#     ''' initialize model from base template, generating CDC, DBP, EDBP, FDBP, GDBP
#     depending on given N-filter length and trainable parameters

#     Args:
#         data:
#         base_conf: a dict of kwargs to make base module, see `make_base_module`
#         sparams_flatkeys: a list of keys contains the static(nontrainable) parameters.
#             For example, assume base module has parameters represented as nested dict
#             {'color': 'red', 'size': {'width': 1, 'height': 2}}, its flatten layout is dict
#              {('color',): 'red', ('size', 'width',): 1, ('size', 'height'): 2}, a sparams_flatkeys
#              of [('color',): ('size', 'width',)] means 'color' and 'size/width' parameters are static.
#             regexp key is supportted.
#         n_symbols: number of symbols used to initialize model, use the minimal value greater than channel
#             memory
#         sps: sample per symbol. Only integer sps is supported now.

#     Returns:
#         a initialized model wrapped by a namedtuple
#     '''
    
#     mod = make_base_module(**base_conf, w0=data.w0)
#     y0 = data.y[:n_symbols * sps]
#     rng0 = random.PRNGKey(0)
#     z0, v0 = mod.init(rng0, core.Signal(y0))
#     ol = z0.t.start - z0.t.stop
#     sparams, params = util.dict_split(v0['params'], sparams_flatkeys)
#     state = v0['af_state']
#     aux = v0['aux_inputs']
#     const = v0['const']
#     return Model(mod, (params, state, aux, const, sparams), ol, name)

def model_init(data: gdat.Input,
               base_conf: dict,
               sparams_flatkeys: list,
               n_symbols: int = 4000,
               sps: int = 2,
               name: str = 'Model'):
    """
    与原 SOTANN.model_init 等价，只是在 v0 中若缺
    'af_state' / 'aux_inputs' / 'const' 就填空 dict。
    """
    # 1. 构网络骨架
    mod = make_base_module(**base_conf, w0=data.w0)

    # 2. 随机一次 forward 拿到所有 collections
    rng0 = random.PRNGKey(0)
    y0   = data.y[:n_symbols * sps]
    z0, v0 = mod.init(rng0, core.Signal(y0))

    # -------- 补空 dict --------------
    v0_mut = unfreeze(v0)
    for col in ('af_state', 'aux_inputs', 'const'):
        v0_mut.setdefault(col, {})
    v0 = freeze(v0_mut)
    # ---------------------------------

    # 3. 拆可训练 / 静态参数
    ol       = z0.t.start - z0.t.stop
    sparams, params = util.dict_split(v0['params'], sparams_flatkeys)

    # 4. 封装 Model
    state = v0['af_state']
    aux   = v0['aux_inputs']
    const = v0['const']
    return Model(mod, (params, state, aux, const, sparams), ol, name)

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

    # 0/1 masks — 必须是 float32 / float64，不能用 complex
    m_in  = (r < thr_in).astype(jnp.float32)
    m_mid = ((r >= thr_in) & (r < thr_mid)).astype(jnp.float32)
    m_out = (r >= thr_mid).astype(jnp.float32)

    def _evm(mask):
        num = jnp.sum(err2 * mask)
        den = jnp.sum(sig2 * mask)
        den = jnp.where(den < 1e-8, 1e-8, den)  # ★ 护栏
        return num / den

    return (w_in  * _evm(m_in) +
            w_mid * _evm(m_mid) +
            w_out * _evm(m_out))

        
def si_snr_flat_amp_pair(tx, rx, eps=1e-8):
    """幅度|·|，两极化展平后算 α，再平分给两路"""
    s  = jnp.reshape(jnp.abs(tx), (-1,))
    x  = jnp.reshape(jnp.abs(rx), (-1,))
    alpha = jnp.vdot(s, x).real / (jnp.vdot(s, s).real + eps)
    e  = x - alpha * s
    snr_db = 10. * jnp.log10( (jnp.vdot(alpha*s, alpha*s).real + eps) /
                              (jnp.vdot(e, e).real + eps) )
    return -snr_db                   

CONST_16QAM = jnp.array([
    -3-3j, -3-1j, -3+3j, -3+1j,
    -1-3j, -1-1j, -1+3j, -1+1j,
     3-3j,  3-1j,  3+3j,  3+1j,
     1-3j,  1-1j,  1+3j,  1+1j
], dtype=jnp.complex64) / jnp.sqrt(10.)

# 2. CE-loss helper  (可 jit / vmap)
def _ce_loss_16qam(pred_sym: Array, true_sym: Array) -> Array:
    """
    pred_sym : [N] complex  — 网络输出符号
    true_sym : [N] complex  — 对齐后的发送符号
    return    : 标量 cross-entropy 损失
    """
    # logits =  –|y − s_k|²   (欧氏距离越小 → logit 越大)
    logits = -jnp.square(jnp.abs(pred_sym[..., None] - CONST_16QAM))   # [N,16]

    # 每个真实符号对应的 QAM 点下标
    label_idx = jnp.argmin(
        jnp.square(jnp.abs(true_sym[..., None] - CONST_16QAM)),
        axis=-1)                                                      # [N]

    # softmax-cross-entropy:
    log_prob = logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True)
    ce = -jnp.take_along_axis(log_prob, label_idx[..., None], axis=-1).squeeze(-1)

    return ce.mean()

_bits = (
    (0,0,0,0), (0,0,0,1), (0,0,1,0), (0,0,1,1),
    (0,1,0,0), (0,1,0,1), (0,1,1,0), (0,1,1,1),
    (1,0,0,0), (1,0,0,1), (1,0,1,0), (1,0,1,1),
    (1,1,0,0), (1,1,0,1), (1,1,1,0), (1,1,1,1)
)
BIT_MAP = jnp.array(_bits, dtype=jnp.float32)  # [16,4]
# 每 bit 的权重向量，顺序 = [b3(MSB), b2, b1, b0(LSB)]
BIT_WEIGHTS = jnp.array([1.2, 1.0, 1.0, 0.8], dtype=jnp.float32)

def _bit_bce_loss_16qam(pred_sym: Array, true_sym: Array) -> Array:
    logits  = -jnp.square(jnp.abs(pred_sym[..., None] - CONST_16QAM))   # [N,16]
    logp    = logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True) # [N,16]
    probs   = jnp.exp(logp)                                             # [N,16]

    # bit 概率 & 真值
    p1 = (probs @ BIT_MAP)                    # P(bit=1)
    p0 = 1.0 - p1

    idx  = jnp.argmin(jnp.square(jnp.abs(true_sym[..., None] - CONST_16QAM)), axis=-1)
    bits = BIT_MAP[idx]                       # 真值 bits

    bce  = -(bits * jnp.log(p1+1e-12) + (1.-bits) * jnp.log(p0+1e-12))  # [N,4]
    return (bce * BIT_WEIGHTS).mean()          # 加权平均
        


def _slice(sig: Signal, start: int, stop: int) -> Signal:
    """返回 sig 在全局样本区间 [start, stop) 的切片。"""
    x, t = sig
    # 把全局坐标映射到数组内部下标
    x_slice = x[start - t.start : x.shape[0] + (stop - t.stop)]
    return Signal(x_slice, SigTime(start, stop, t.sps))

def robust_align(a: Signal, b_raw):
    """
    让网络输出 `a` 与真值 `b_raw` 在样本维度上对齐。
    支持 `b_raw` 为:
      • Signal (带/不带有效时间轴) • 裸 ndarray
    总能返回 **长度一致** 的 `(a_aligned, b_aligned)`。
    """
    # -------- 把 b_raw 也包装成 Signal ---------------------------------
    if isinstance(b_raw, Signal):
        b = b_raw
    else:  # ndarray
        # 如果没时间轴，就先用 (0,0) 占位
        b = Signal(b_raw, SigTime(0, 0, a.t.sps))

    # -------- 计算重叠区间 ---------------------------------------------
    start = max(a.t.start, b.t.start)
    stop  = min(a.t.stop , b.t.stop )
    # 若没有交集就按最短长度裁剪
    if start >= stop:
        # 取两端都能覆盖的最小长度
        N = min(a.val.shape[0], b.val.shape[0])
        start_a = a.t.start
        start_b = b.t.start
        a = _slice(a, start_a, start_a + N)
        b = _slice(b, start_b, start_b + N)
        return a, b

    # 有交集 → 直接裁剪到重叠区
    return _slice(a, start, stop), _slice(b, start, stop)

def loss_fn(module : layer.Layer,
            params : Dict,
            state  : Dict,
            y      : core.Array,
            x      : core.Array,
            aux    : Dict,
            const  : Dict,
            sparams: Dict,
            β_ce   : float = 0.5,
            λ_kl   : float = 1e-4):
    """训练损失：SNR+EVM + 可选 BCE + IB‑KL（含对齐鲁棒性）"""
    # ---- 前向 ------------------------------------------------------------------
    params_net = util.dict_merge(params, sparams)
    z_out, state_new = module.apply(
        {'params': params_net,
         'aux_inputs': aux,
         'const': const, **state},
        core.Signal(y)
    )

    # ---- 与真值对齐 ------------------------------------------------------------
    z_align, x_align = robust_align(z_out, x)      # <= 关键改动

    # ---- 1) SNR + EVM ----------------------------------------------------------
    snr = si_snr_flat_amp_pair(jnp.abs(z_align.val), jnp.abs(x_align.val))
    evm = evm_ring(jnp.abs(z_align.val), jnp.abs(x_align.val))
    loss_main = snr + 0.1 * evm

    # ---- 2) Bit‑wise BCE -------------------------------------------------------
    bit_bce = _bit_bce_loss_16qam(z_align.val, x_align.val)
    loss_main += β_ce * bit_bce

    # ---- 3) Information‑Bottleneck KL -----------------------------------------
    kl_ib = 0.5 * jnp.mean(jnp.square(jnp.abs(z_align.val)))
    total_loss = loss_main + λ_kl * kl_ib

    return total_loss, state_new
              
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
    ''' single backprop step

        Args:
            model: model returned by `model_init`
            opt: optimizer
            i: iteration counter
            opt_state: optimizer state
            module_state: module state
            y: transmitted waveforms
            x: aligned sent symbols
            aux: auxiliary input
            const: contants (internal info generated by model)
            sparams: static parameters

        Return:
            loss, updated module state
    '''

    params = opt.params_fn(opt_state)
    (loss, module_state), grads = value_and_grad(
        loss_fn, argnums=1, has_aux=True)(module, params, module_state, y, x,
                                          aux, const, sparams)
    opt_state = opt.update_fn(i, grads, opt_state)
    return loss, opt_state, module_state
                  
                        
def get_train_batch(ds: gdat.Input,
                    batchsize: int,
                    overlaps: int,
                    sps: int = 2):
    ''' generate overlapped batch input for training

        Args:
            ds: dataset
            batchsize: batch size in symbol unit
            overlaps: overlaps in symbol unit
            sps: samples per symbol

        Returns:
            number of symbols,
            zipped batched triplet input: (recv, sent, fomul)
    '''

    flen = batchsize + overlaps
    fstep = batchsize
    ds_y = op.frame_gen(ds.y, flen * sps, fstep * sps)
    ds_x = op.frame_gen(ds.x, flen, fstep)
    n_batches = op.frame_shape(ds.x.shape, flen, fstep)[0]
    return n_batches, zip(ds_y, ds_x)

def train(model: Model,
          data: gdat.Input,
          batch_size: int = 500,
          n_iter = None,
          opt: optim.Optimizer = optim.adam(optim.piecewise_constant([500, 1000], [1e-4, 1e-5, 1e-6]))):
    ''' training process (1 epoch)

        Args:
            model: Model namedtuple return by `model_init`
            data: dataset
            batch_size: batch size
            opt: optimizer

        Returns:
            yield loss, trained parameters, module state
    '''

    params, module_state, aux, const, sparams = model.initvar
    opt_state = opt.init_fn(params)

    n_batch, batch_gen = get_train_batch(data, batch_size, model.overlaps)
    n_iter = n_batch if n_iter is None else min(n_iter, n_batch)

    for i, (y, x) in tqdm(enumerate(batch_gen),
                             total=n_iter, desc='training', leave=False):
        if i >= n_iter: break
        aux = core.dict_replace(aux, {'truth': x})
        loss, opt_state, module_state = update_step(model.module, opt, i, opt_state,
                                                   module_state, y, x, aux,
                                                   const, sparams)
        yield loss, opt.params_fn(opt_state), module_state
                                
                       
def test(model: Model,
         params: Dict,
         data: gdat.Input,
         eval_range: tuple=(300000, -20000),
         metric_fn=comm.qamqot):
    ''' testing, a simple forward pass

        Args:
            model: Model namedtuple return by `model_init`
        data: dataset
        eval_range: interval which QoT is evaluated in, assure proper eval of steady-state performance
        metric_fn: matric function, comm.snrstat for global & local SNR performance, comm.qamqot for
            BER, Q, SER and more metrics.

        Returns:
            evaluated matrics and equalized symbols
    '''

    state, aux, const, sparams = model.initvar[1:]
    aux = core.dict_replace(aux, {'truth': data.x})
    if params is None:
      params = model.initvar[0]

    z, _ = jit(model.module.apply,
               backend='cpu')({
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

                
def equalize_dataset(model_te, params, state_bundle, data):
    module_state, aux, const, sparams = state_bundle
    z,_ = jax.jit(model_te.module.apply, backend='cpu')(
        {'params': util.dict_merge(params, sparams),
         'aux_inputs': aux, 'const': const, **module_state},
        core.Signal(data.y))

    start, stop = z.t.start, z.t.stop
    z_eq  = np.asarray(z.val[:,0])          # equalized
    s_ref = np.asarray(data.x)[start:stop,0]   # 保持原尺度
    return z_eq, s_ref

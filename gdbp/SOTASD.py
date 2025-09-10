# [np.float64(8.488569856531333), np.float64(8.810227272425061)]
# [np.float64(0.0039393303488155194), np.float64(0.0029124958132475207)]
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
Model = namedtuple('Model', 'module initvar overlaps name')
Array = Any
Dict = Union[dict, flax.core.FrozenDict]


## Two ##
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

    # 定义串联的 FDBP 层
    fdbp_series = layer.Serial(
        layer.FDBP(steps=steps,
                    dtaps=dtaps,
                    ntaps=ntaps,
                    d_init=d_init,
                    n_init=n_init,
                    name='fdbp1'),
        layer.BatchPowerNorm(mode=mode),
        layer.MIMOFOEAf(name='FOEAf1',
                        w0=w0,
                        train=mimo_train,
                        preslicer=core.conv1d_slicer(rtaps),
                        foekwargs={}),
        layer.vmap(layer.Conv1d)(name='RConv1', taps=rtaps),
        layer.MIMOAF(train=mimo_train),
        name='fdbp_series'
    )

    # 定义原有的串行分支
    serial_branch = layer.Serial(
        layer.FDBP1(steps=steps,
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
        layer.vmap(layer.Conv1d)(name='RConv', taps=rtaps),
        layer.MIMOAF(train=mimo_train),
        name='serial_branch'  # 添加名称
    )

    # 定义基础模块
    base = layer.Serial(
        layer.FanOut(num=2),
        layer.Parallel(
            fdbp_series,
            serial_branch
        ),
        layer.FanInMean()
    )

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


def model_init(data: gdat.Input,
               base_conf: dict,
               sparams_flatkeys: list,
               n_symbols: int = 4000,
               sps : int = 2,
               name='Model'):
    ''' initialize model from base template, generating CDC, DBP, EDBP, FDBP, GDBP
    depending on given N-filter length and trainable parameters

    Args:
        data:
        base_conf: a dict of kwargs to make base module, see `make_base_module`
        sparams_flatkeys: a list of keys contains the static(nontrainable) parameters.
            For example, assume base module has parameters represented as nested dict
            {'color': 'red', 'size': {'width': 1, 'height': 2}}, its flatten layout is dict
             {('color',): 'red', ('size', 'width',): 1, ('size', 'height'): 2}, a sparams_flatkeys
             of [('color',): ('size', 'width',)] means 'color' and 'size/width' parameters are static.
            regexp key is supportted.
        n_symbols: number of symbols used to initialize model, use the minimal value greater than channel
            memory
        sps: sample per symbol. Only integer sps is supported now.

    Returns:
        a initialized model wrapped by a namedtuple
    '''
    
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

def align_phase_scale(yhat: jnp.ndarray, y: jnp.ndarray, eps: float = 1e-8):
    """
    将预测 yhat 沿 U(1)×R+ 对齐到参考 y：
    min_{phi, alpha} || alpha * e^{-j phi} yhat - y ||^2
    返回对齐后的 yhat' 以及 (alpha, phi) 便于记录。
    """
    z = jnp.vdot(yhat, y)                          # <yhat, y>
    p = z / (jnp.abs(z) + eps)                     # 单位相位子（避免 atan2 奇异）
    yhp = yhat * jnp.conj(p)                       # 去相位：e^{-j phi} = conj(p)
    alpha = jnp.real(jnp.vdot(yhp, y)) / (jnp.real(jnp.vdot(yhp, yhp)) + eps)
    return alpha * yhp, alpha, jnp.angle(p)
  
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
        

def loss_fn(module, params, state, y, x, aux, const, sparams,
            β_ce: float = 0.5, λ_kl: float = 1e-4):

    params_net = util.dict_merge(params, sparams)
    z_out, state_new = module.apply(
        {'params': params_net, 'aux_inputs': aux, 'const': const, **state},
        core.Signal(y)
    )

    x_ref = x[z_out.t.start:z_out.t.stop]   # 参考符号
    yhat  = z_out.val                       # 预测复符号

    # --- 仅移除相位（商空间代表），并 stop-gradient ---
    zc   = jnp.vdot(yhat.reshape(-1), x_ref.reshape(-1))   # <ŷ, x>
    p    = zc / (jnp.abs(zc) + 1e-8)                       # e^{jφ*}
    p    = lax.stop_gradient(p)                            # ★ 不反传 φ*
    y_al = yhat * jnp.conj(p)                              # 去相位后的输出
    # 不做幅度对齐（alpha）；把幅度学习留给 BN/MIMOAF

    # --- 任务一致计分（在 y_al 上）---
    # EVM（Q²代理）
    evm = (jnp.mean(jnp.abs(y_al - x_ref)**2) /
           (jnp.mean(jnp.abs(x_ref)**2) + 1e-8))

    # SI-SNR（复数版）
    t = x_ref.reshape(-1); e = y_al.reshape(-1)
    a = jnp.vdot(t, e) / (jnp.vdot(t, t) + 1e-8)
    s = a * t
    snr = -10.0 * jnp.log10(
        (jnp.real(jnp.vdot(s, s)) + 1e-8) /
        (jnp.real(jnp.vdot(e - s, e - s)) + 1e-8)
    )

    # Bit-BCE（对齐后星座）
    bit_bce = _bit_bce_loss_16qam(y_al, x_ref)

    # 信息瓶颈 KL（保留在未对齐输出上，限制能量）
    kl_ib = 0.5 * jnp.mean(jnp.square(jnp.abs(yhat)))

    loss_main  = snr + 0.1 * evm + β_ce * bit_bce
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

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
Model = namedtuple('Model', 'module initvar overlaps name')
Array = Any
Dict = Union[dict, flax.core.FrozenDict]


from flax import linen as flnn
from flax.core import freeze  # 后面 model_init 也会用到

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

    # —— 关键点：保留 FOEAf（含 preslicer）与 MIMOAF 原本的命名与结构 ——
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
        name='serial_branch'
    )

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


from flax.core import freeze


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

    # 这些集合有的版本可能不存在 → 用空 dict 兜底
    af_state   = v0.get('af_state',   freeze({}))
    aux_inputs = v0.get('aux_inputs', freeze({}))
    const      = v0.get('const',      freeze({}))

    # apply(**state) 需要“顶层集合字典”
    state = {'af_state': af_state}

    return Model(mod, (params, state, aux_inputs, const, sparams), ol, name)





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
        

def _crop_flat_pair(yhat, x_ref):
    # yhat: [...], x_ref: [...]
    yv = jnp.ravel(yhat)
    xv = jnp.ravel(x_ref)
    L = min(yv.shape[0], xv.shape[0])  # 静态shape，JIT友好
    return yv[:L], xv[:L]

def _loss_qal_core(yhat, x_ref):
    """QAL：去全局相位（stop-grad），不做幅度对齐"""
    yv, xv = _crop_flat_pair(yhat, x_ref)

    # 相位闭式 + stop-grad（不反传相位）
    zc = jnp.vdot(yv, xv)                                   # <ŷ, x>
    p  = lax.stop_gradient(zc / (jnp.abs(zc) + 1e-8))       # e^{jφ*}
    yal = yv * jnp.conj(p)                                   # 去相位后的输出

    # 相位不敏感的任务一致性：EVM、复SI-SNR(在去相位yal上)、Bit-BCE
    evm = jnp.mean(jnp.abs(yal - xv)**2) / (jnp.mean(jnp.abs(xv)**2) + 1e-8)

    a   = jnp.vdot(xv, yal) / (jnp.vdot(xv, xv) + 1e-8)
    s   = a * xv
    snr = -10.0 * jnp.log10(
        (jnp.real(jnp.vdot(s, s)) + 1e-8) /
        (jnp.real(jnp.vdot(yal - s, yal - s)) + 1e-8)
    )

    bit_bce = _bit_bce_loss_16qam(yal, xv)
    return snr, evm, bit_bce

def _loss_plain_core(yhat, x_ref):
    """Plain：不做任何相位/幅度对齐——对相位敏感"""
    yv, xv = _crop_flat_pair(yhat, x_ref)

    evm = jnp.mean(jnp.abs(yv - xv)**2) / (jnp.mean(jnp.abs(xv)**2) + 1e-8)

    a   = jnp.vdot(xv, yv) / (jnp.vdot(xv, xv) + 1e-8)
    s   = a * xv
    snr = -10.0 * jnp.log10(
        (jnp.real(jnp.vdot(s, s)) + 1e-8) /
        (jnp.real(jnp.vdot(yv - s, yv - s)) + 1e-8)
    )

    bit_bce = _bit_bce_loss_16qam(yv, xv)
    return snr, evm, bit_bce

def loss_fn(module: layer.Layer,
            params: Dict,
            state : Dict,
            y     : Array,
            x     : Array,
            aux   : Dict,
            const : Dict,
            sparams: Dict,
            β_ce : float = 0.5,
            λ_kl : float = 1e-4,
            use_qal: bool = True):   # ★ 新增

    params_net = util.dict_merge(params, sparams)
    z_out, state_new = module.apply(
        {'params': params_net, 'aux_inputs': aux, 'const': const, **state},
        core.Signal(y)
    )
    x_ref = x[z_out.t.start:z_out.t.stop]

    if use_qal:
        snr, evm, bit_bce = _loss_qal_core(z_out.val, x_ref)
    else:
        snr, evm, bit_bce = _loss_plain_core(z_out.val, x_ref)

    # IB-KL：限制幅度能量（用未对齐输出）
    kl_ib = 0.5 * jnp.mean(jnp.square(jnp.abs(jnp.ravel(z_out.val))))

    loss_main  = snr + 0.1 * evm + β_ce * bit_bce
    total_loss = loss_main + λ_kl * kl_ib
    return total_loss, state_new
              
@partial(jit, backend='cpu', static_argnums=(0, 1, 12))
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
                β_ce: float = 0.5,
                λ_kl: float = 1e-4,
                use_qal: bool = True):  # ★ 新增

    params = opt.params_fn(opt_state)
    (loss, module_state), grads = value_and_grad(
        loss_fn, argnums=1, has_aux=True)(
            module, params, module_state, y, x, aux, const, sparams, β_ce, λ_kl, use_qal
        )
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
    params, module_state, aux, const, sparams = model.initvar
    opt_state = opt.init_fn(params)

    n_batch, batch_gen = get_train_batch(data, batch_size, model.overlaps)
    n_iter = n_batch if n_iter is None else min(n_iter, n_batch)

    for i, (y, x) in tqdm(enumerate(batch_gen), total=n_iter, desc='training', leave=False):
        if i >= n_iter: break
        # ★ 不要 teacher-forcing（不要把 truth 喂给 aux）
        # aux = core.dict_replace(aux, {'truth': x})
        loss, opt_state, module_state = update_step(
            model.module, opt, i, opt_state, module_state, y, x, aux, const, sparams
        )
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

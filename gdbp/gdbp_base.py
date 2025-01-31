from jax import numpy as jnp, random, jit, value_and_grad, nn
import flax
from commplax import util, comm, cxopt, op, optim
from commplax.module import core, layer
import numpy as np
from functools import partial
from collections import namedtuple
from tqdm.auto import tqdm
from typing import Any, Optional, Union
from . import data as gdat
import jax
from scipy import signal
from flax import linen as nn
from sklearn.neighbors import KernelDensity
from jax.scipy.stats import norm
from jax import jit, lax
Model = namedtuple('Model', 'module initvar overlaps name')
Array = Any
Dict = Union[dict, flax.core.FrozenDict]

## One ##
def make_base_module(steps: int = 3,
                     dtaps: int = 261,
                     ntaps: int = 41,
                     rtaps: int = 61,
                     init_fn: tuple = (core.delta, core.gauss),
                     w0 = 0.,
                     mode: str = 'train'):
    '''
    make base module that derives DBP, FDBP, EDBP, GDBP depending on
    specific initialization method and trainable parameters defined
    by trainer.

    Args:
        steps: GDBP steps/layers
        dtaps: D-filter length
        ntaps: N-filter length
        rtaps: R-filter length
        init_fn: a tuple contains a pair of initializer for D-filter and N-filter
        mode: 'train' or 'test'

    Returns:
        A layer object
    '''

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
        layer.BatchPowerNorm(mode=mode),
        layer.MIMOFOEAf(name='FOEAf',
                        w0=w0,
                        train=mimo_train,
                        preslicer=core.conv1d_slicer(rtaps),
                        foekwargs={}),
        layer.vmap(layer.Conv1d)(name='RConv', taps=rtaps),  # vectorize column-wise Conv1D
        layer.MIMOAF(train=mimo_train))  # adaptive MIMO layer
        
    return base

## Two ##
# def make_base_module(steps: int = 3,
#                      dtaps: int = 261,
#                      ntaps: int = 41,
#                      rtaps: int = 61,
#                      init_fn: tuple = (core.delta, core.gauss),
#                      w0=0.,
#                      mode: str = 'train'):

#     _assert_taps(dtaps, ntaps, rtaps)

#     d_init, n_init = init_fn

#     if mode == 'train':
#         mimo_train = True
#     elif mode == 'test':
#         mimo_train = cxopt.piecewise_constant([200000], [True, False])
#     else:
#         raise ValueError('invalid mode %s' % mode)

#     # 定义串联的 FDBP 层
#     fdbp_series = layer.Serial(
#         layer.FDBP(steps=steps,
#                     dtaps=dtaps,
#                     ntaps=ntaps,
#                     d_init=d_init,
#                     n_init=n_init,
#                     name='fdbp1'),
#         layer.BatchPowerNorm(mode=mode),
#         layer.MIMOFOEAf(name='FOEAf1',
#                         w0=w0,
#                         train=mimo_train,
#                         preslicer=core.conv1d_slicer(rtaps),
#                         foekwargs={}),
#         layer.vmap(layer.Conv1d)(name='RConv1', taps=rtaps),
#         layer.MIMOAF(train=mimo_train),
#         name='fdbp_series'
#     )

#     # 定义原有的串行分支
#     serial_branch = layer.Serial(
#         layer.FDBP1(steps=steps,
#                    dtaps=dtaps,
#                    ntaps=ntaps,
#                    d_init=d_init,
#                    n_init=n_init),
#         layer.BatchPowerNorm1(mode=mode),
#         layer.MIMOFOEAf1(name='FOEAf',
#                         w0=w0,
#                         train=mimo_train,
#                         preslicer=core.conv1d_slicer(rtaps),
#                         foekwargs={}),
#         layer.vmap(layer.Conv1d1)(name='RConv', taps=rtaps),
#         layer.MIMOAF1(train=mimo_train),
#         name='serial_branch'  # 添加名称
#     )

#     # 定义基础模块
#     base = layer.Serial(
#         layer.FanOut(num=2),
#         layer.Parallel(
#             fdbp_series,
#             serial_branch
#         ),
#         layer.FanInMean()
#     )

#     return base

## Three ##
# def make_base_module(steps: int = 3,
#                      dtaps: int = 261,
#                      ntaps: int = 41,
#                      rtaps: int = 61,
#                      init_fn: tuple = (core.delta, core.gauss),
#                      w0=0.,
#                      mode: str = 'train'):

#     _assert_taps(dtaps, ntaps, rtaps)

#     d_init, n_init = init_fn

#     if mode == 'train':
#         mimo_train = True
#     elif mode == 'test':
#         mimo_train = cxopt.piecewise_constant([200000], [True, False])
#     else:
#         raise ValueError('invalid mode %s' % mode)

#     # first tree
#     fdbp_series = layer.Serial(
#         layer.FDBP(steps=steps,
#                    dtaps=dtaps,
#                    ntaps=ntaps,
#                    d_init=d_init,
#                    n_init=n_init,
#                    name='fdbp1'),
#         layer.BatchPowerNorm(mode=mode),
#         layer.MIMOFOEAf(name='FOEAf1',
#                         w0=w0,
#                         train=mimo_train,
#                         preslicer=core.conv1d_slicer(rtaps),
#                         foekwargs={}),
#         layer.vmap(layer.Conv1d)(name='RConv1', taps=rtaps),
#         layer.MIMOAF(train=mimo_train),
#         name='fdbp_series'
#     )

#     # second tree
#     serial_branch = layer.Serial(
#         layer.FDBP1(steps=steps,
#                     dtaps=dtaps,
#                     ntaps=ntaps,
#                     d_init=d_init,
#                     n_init=n_init),
#         layer.BatchPowerNorm(mode=mode),
#         layer.MIMOFOEAf(name='FOEAf',
#                         w0=w0,
#                         train=mimo_train,
#                         preslicer=core.conv1d_slicer(rtaps),
#                         foekwargs={}),
#         layer.vmap(layer.Conv1d)(name='RConv', taps=rtaps),
#         layer.MIMOAF(train=mimo_train),
#         name='serial_branch'
#     )

#     # third three
#     another_branch = layer.Serial(
#         layer.FDBP2(steps=steps,
#                     dtaps=dtaps,
#                     ntaps=ntaps,
#                     d_init=d_init,
#                     n_init=n_init,
#                     name='fdbp2'),
#         layer.BatchPowerNorm1(mode=mode),
#         layer.MIMOFOEAf1(name='FOEAf2',
#                         w0=w0,
#                         train=mimo_train,
#                         preslicer=core.conv1d_slicer(rtaps),
#                         foekwargs={}),
#         layer.vmap(layer.Conv1d1)(name='RConv2', taps=rtaps),
#         layer.MIMOAF1(train=mimo_train),
#         name='another_branch'
#     )
#     base = layer.Serial(
#         layer.FanOut(num=3),
#         layer.Parallel(
#             fdbp_series,
#             serial_branch,
#             another_branch
#         ),
#         layer.FanInMean()
#     )

#     return base



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
  
def negative_cosine_similarity(p, z):
    p = l2_normalize(p, axis=1)
    z = l2_normalize(z, axis=1)
    return -jnp.mean(jnp.sum(p * z, axis=1))
  
#scale-1
def apply_transform(x, scale_range=(0.5, 2.0), p=0.5):
    if np.random.rand() < p:
        scale = np.random.uniform(scale_range[0], scale_range[1])
        x = x * scale
    return x
  
# shift-2  
def apply_transform1(x, shift_range=(-5.0, 5.0), p=0.5):
    if np.random.rand() < p:
        shift = np.random.uniform(shift_range[0], shift_range[1])
        x = x + shift
    return x
  
# mask-3
# def apply_transform2(x, range=(0, 300), p=0.5):
#     if np.random.rand() < p:
#         mask_len = int(np.random.uniform(range[0], range[1]))
#         start = int(np.random.uniform(0, len(x) - mask_len))
#         mask = jnp.ones_like(x)
#         mask = mask.at[start:start + mask_len].set(0)
#         x = x * mask
#     return x

# def apply_transform2(x, p=0.5):
#     if np.random.rand() < p:
#         total_length = x.shape[0]
#         mask = np.random.choice([0, 1], size=total_length, p=[1-p, p])
#         mask = jnp.array(mask)
#         x = x * mask
#     return x

def apply_transform2(x, mask_range=(0, 300), p=0.8):
    if np.random.rand() < p:
        total_length = x.shape[0]
        mask = np.random.choice([0, 1], size=total_length, p=[1-p, p])
        mask = jnp.array(mask)[:, None]
        mask = jnp.broadcast_to(mask, x.shape)
        x = x * mask
    return x
  
# random-4
def apply_transform3(x, range=(0.0, 0.2), p=0.5):
    if np.random.rand() < p:
        sigma = np.random.uniform(range[0], range[1])
        x = x + np.random.normal(0, sigma, x.shape)
    return x
  
def apply_combined_transform(x, scale_range=(0.5, 2.0), shift_range=(-5.0, 5.0), shift_range1=(-100, 100), mask_range=(0, 30), noise_range=(0.0, 0.2), p_scale=0.5, p_shift=0.5, p_mask=0.5, p_noise=0.5, p=0.5):
    if np.random.rand() < p_scale:
        scale = np.random.uniform(scale_range[0], scale_range[1])
        x = x * scale

    if np.random.rand() < p_shift:
        shift = np.random.uniform(shift_range[0], shift_range[1])
        x = x + shift

    if np.random.rand() < p_mask:
        total_length = x.shape[0]
        mask = np.random.choice([0, 1], size=total_length, p=[1-p_mask, p_mask])
        mask = jnp.array(mask)[:, None]
        mask = jnp.broadcast_to(mask, x.shape)
        x = x * mask

    if np.random.rand() < p_noise:
        sigma = np.random.uniform(noise_range[0], noise_range[1])
        noise = np.random.normal(0, sigma, x.shape)
        x = x + noise

    if np.random.rand() < p:
        t_shift = np.random.randint(shift_range1[0], shift_range1[1])
        x = jnp.roll(x, shift=t_shift)
    return x
  
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
import jax
import jax.numpy as jnp

def safe_logsumexp(values: jnp.ndarray,
                   mask: jnp.ndarray,
                   eps: float = 1e-9) -> jnp.ndarray:
    """
    针对已做掩码的子集，对其做 log-sum-exp，避免直接 exp() 加和造成上溢/下溢。
    
    :param values: shape (16,). 通常是 e_k = -dist_k^2/(2*sigma^2)
    :param mask:   shape (16,). 布尔掩码, 表示只累加部分星座点
    :param eps:    防止 log(0)
    :return:       标量，对应 log( sum( exp(values) )) with mask applied
    """
    neg_inf = jnp.array(-1e20, dtype=values.dtype)
    # 对不需要的分量设为 -∞
    masked_vals = jnp.where(mask, values, neg_inf)
    max_val = jnp.max(masked_vals)
    # 如果全是 -∞，max_val 即 -∞；给它一个保护
    max_val = jnp.where(jnp.isfinite(max_val), max_val, 0.0)
    # log-sum-exp
    return max_val + jnp.log(jnp.sum(jnp.exp(masked_vals - max_val) + eps))

def two_bits_to_level(b0, b1):
    """
    b0, b1 ∈ {0,1}，返回 Gray-code 下的一维电平 ∈ {-3, -1, +1, +3}.
    约定: 00->-3, 01->-1, 11->+1, 10->+3 (常见 Gray-code 次序)
    """
    # (b0,b1) => index => level
    idx = (b0 << 1) ^ b1  # XOR 确保 Gray 次序
    # idx ∈ {0,1,3,2}
    levels = jnp.array([-3, -1, +3, +1], dtype=jnp.float32)
    return levels[idx]

def make_16qam_constellation():
    """
    枚举 4bits => 16个星座点 (I, Q)，以及对应的比特标签 BIT_LABELS。
    我们定义 bit 顺序为 [b0, b1, b2, b3], 其中:
      - Q轴 = two_bits_to_level(b0,b1)
      - I轴 = two_bits_to_level(b2,b3)
    """
    all_bits = []
    all_points = []
    for b0 in [0,1]:
        for b1 in [0,1]:
            for b2 in [0,1]:
                for b3 in [0,1]:
                    I = two_bits_to_level(b2, b3)
                    Q = two_bits_to_level(b0, b1)
                    all_points.append((I, Q))
                    all_bits.append([b0, b1, b2, b3])

    CONSTELLATION = jnp.array(all_points, dtype=jnp.float32)  # shape (16,2)
    BIT_LABELS     = jnp.array(all_bits,   dtype=jnp.int32)   # shape (16,4)

    return CONSTELLATION, BIT_LABELS

# 生成 16QAM 星座表 (在全局使用)
CONSTELLATION, BIT_LABELS = make_16qam_constellation()

def compute_llrs_16qam_one_symbol(r: jnp.ndarray,
                                  sigma: float,
                                  constellation: jnp.ndarray,
                                  bit_labels: jnp.ndarray,
                                  eps: float = 1e-9) -> jnp.ndarray:
    """
    对单个实维符号 r (shape (2,): [I, Q])，计算 16QAM 的 4 比特 LLR (shape (4,)).
    
    注：如果实际系统是复符号 c = rI + j*rQ，可以先把 c 拆成 (rI, rQ) = (2,).
    """
    # 1) 计算与 16个星座点的距离^2 => shape (16,)
    diff = constellation - r[None, :]  # (16,2) - (1,2) => (16,2)
    dist_sq = jnp.sum(diff * diff, axis=-1)

    # 2) e_k = -dist_sq / (2 sigma^2)
    e = -dist_sq / (2.0 * sigma**2 + eps)

    # 3) 对每个 bit i in {0..3}，做 masked log-sum-exp
    llrs = []
    for i in range(4):
        mask_0 = (bit_labels[:, i] == 0)
        mask_1 = ~mask_0

        ls0 = safe_logsumexp(e, mask_0, eps)  # log( sum_{b_i=0} exp(e_k) )
        ls1 = safe_logsumexp(e, mask_1, eps)  # log( sum_{b_i=1} exp(e_k) )

        llr_i = ls0 - ls1
        llrs.append(llr_i)

    return jnp.stack(llrs, axis=0)  # (4,)

def subcarrier_loss_16qam(r_sc: jnp.ndarray,
                          x_sc: jnp.ndarray,
                          constellation: jnp.ndarray,
                          bit_labels: jnp.ndarray,
                          sigma: float = 0.1,
                          eps: float = 1e-9) -> jnp.ndarray:
    """
    针对单个子载波(符号) (r_sc, x_sc) 计算 -GMI。二者皆 shape=(2,)。

    1) 找到 x_sc 在星座中的下标 idx_x => b_vec(4bits)
    2) 用 compute_llrs_16qam_one_symbol(r_sc, ...) 得到 LLR(4,)
    3) 对真实比特 alpha_i ∈ {+1, -1} 做 "log2(1+exp(-alpha_i * LLR_i))"
    4) 累加得到 -GMI
    """
    # 首先计算 r_sc 与 16 个点的距离(可选, 仅在 debug 时需要)
    diff_r = constellation - r_sc[None, :]
    dist_r = jnp.sum(diff_r**2, axis=-1)  # (16,)

    # 计算 x_sc 在星座中的下标 (如果 x_sc 也是理想点)
    diff_x = constellation - x_sc[None, :]
    dist_x = jnp.sum(diff_x**2, axis=-1)
    idx_x = jnp.argmin(dist_x)  # x_sc 对应的星座下标

    # 真实比特 => shape (4,)
    b_vec = bit_labels[idx_x]

    # 计算接收符号 r_sc 的 LLR => (4,)
    llrs = compute_llrs_16qam_one_symbol(r_sc, sigma, constellation, bit_labels, eps)

    # bit-level 损失: -GMI_i ~ log2(1 + exp(- alpha_i * LLR_i))
    alpha = 2.0 * b_vec.astype(jnp.float32) - 1.0  # ∈ {+1, -1}
    bit_losses = jnp.log1p(jnp.exp(-alpha * llrs)) / jnp.log(2.0)
    return jnp.sum(bit_losses)  # 4 bits 累加

def loss_per_sample_16qam(r: jnp.ndarray,
                          x: jnp.ndarray,
                          constellation: jnp.ndarray,
                          bit_labels: jnp.ndarray,
                          sigma: float = 0.1) -> jnp.ndarray:
    """
    处理单个样本，但样本内含 n_sub 个子载波/符号:
    r.shape = (n_sub, 2), x.shape = (n_sub,2).
    """
    # 对 n_sub 逐符号计算 -GMI
    def per_subcarrier(r_sc, x_sc):
        return subcarrier_loss_16qam(r_sc, x_sc, constellation, bit_labels, sigma)

    losses = jax.vmap(per_subcarrier, in_axes=(0,0))(r, x)  # => (n_sub,)
    return jnp.mean(losses)

def gmi_loss_16qam(pred_batch: jnp.ndarray,
                   true_batch: jnp.ndarray,
                   constellation: jnp.ndarray = CONSTELLATION,
                   bit_labels: jnp.ndarray   = BIT_LABELS,
                   sigma: float = 0.1) -> jnp.ndarray:
    """
    最外层封装，对批次维度做 vmap:
    - pred_batch.shape = (B, n_sub, 2)
    - true_batch.shape = (B, n_sub, 2)

    返回标量 (batch 平均的 -GMI)
    """
    # vmap 到 loss_per_sample_16qam
    batch_losses = jax.vmap(
        loss_per_sample_16qam,
        in_axes=(0,0,None,None,None)
    )(pred_batch, true_batch, constellation, bit_labels, sigma)
    return jnp.mean(batch_losses)


def loss_fn(module: layer.Layer,
            params: Dict,
            state: Dict,
            y: Array,
            x: Array,
            aux: Dict,
            const: Dict,
            sparams: Dict,):
    params = util.dict_merge(params, sparams)
    z_original, updated_state = module.apply(
        {'params': params, 'aux_inputs': aux, 'const': const, **state}, core.Signal(y)) 
    # y_transformed = apply_combined_transform(y)
    aligned_x = x[z_original.t.start:z_original.t.stop]
    mse_loss = jnp.mean(jnp.abs(z_original.val - aligned_x) ** 2)
    snr = si_snr(jnp.abs(z_original.val), jnp.abs(aligned_x)) 
    pred_symbol = jnp.stack([jnp.real(z_original.val), jnp.imag(z_original.val)], axis=-1)
    true_symbol = jnp.stack([jnp.real(aligned_x),       jnp.imag(aligned_x)],       axis=-1)

    gmi_loss_val = gmi_loss_16qam(
        pred_symbol, 
        true_symbol,
        sigma         = 0.1,
        constellation = CONSTELLATION,
        bit_labels    = BIT_LABELS
    )
    return gmi_loss_val, updated_state


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
          opt: optim.Optimizer = optim.sgd(optim.piecewise_constant([500, 1000], [1e-4, 1e-5, 1e-6]))):
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

# def test(model: Model,
#          params: Dict,
#          data: gdat.Input,
#          eval_range: tuple=(300000, -20000),
#          metric_fn=comm.qamqot):
#     ''' testing, a simple forward pass '''

#     state, aux, const, sparams = model.initvar[1:]
#     aux = core.dict_replace(aux, {'truth': data.x})
#     if params is None:
#         params = model.initvar[0]

#     z, _ = jit(model.module.apply,
#                backend='cpu')({
#                    'params': util.dict_merge(params, sparams),
#                    'aux_inputs': aux,
#                    'const': const,
#                    **state
#                }, core.Signal(data.y))

#     # 拆分拼接的输出
#     output_dbp, output_nn = jnp.split(z.val, indices_or_sections=2, axis=-1)

#     # 选择要用于计算 Q 值的输出，例如 DBP 分支的输出
#     output = output_dbp.squeeze()

#     # 对齐原始信号
#     aligned_x = data.x[z.t.start:z.t.stop]

#     # 确保输出和原始信号形状一致
#     output = output[:aligned_x.shape[0]]

#     # 计算指标
#     metric = metric_fn(output,
#                        aligned_x,
#                        scale=np.sqrt(10),
#                        eval_range=eval_range)
#     return metric, z

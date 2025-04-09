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
# def make_base_module(steps: int = 3,
#                      dtaps: int = 261,
#                      ntaps: int = 41,
#                      rtaps: int = 61,
#                      init_fn: tuple = (core.delta, core.gauss),
#                      w0 = 0.,
#                      mode: str = 'train'):
#     '''
#     make base module that derives DBP, FDBP, EDBP, GDBP depending on
#     specific initialization method and trainable parameters defined
#     by trainer.

#     Args:
#         steps: GDBP steps/layers
#         dtaps: D-filter length
#         ntaps: N-filter length
#         rtaps: R-filter length
#         init_fn: a tuple contains a pair of initializer for D-filter and N-filter
#         mode: 'train' or 'test'

#     Returns:
#         A layer object
#     '''

#     _assert_taps(dtaps, ntaps, rtaps)

#     d_init, n_init = init_fn

#     if mode == 'train':
#         # configure mimo to its training mode
#         mimo_train = True
#     elif mode == 'test':
#         # mimo operates at training mode for the first 200000 symbols,
#         # then switches to tracking mode afterwards
#         mimo_train = cxopt.piecewise_constant([200000], [True, False])
#     else:
#         raise ValueError('invalid mode %s' % mode)
        
#     base = layer.Serial(
#         layer.FDBP(steps=steps,
#                    dtaps=dtaps,
#                    ntaps=ntaps,
#                    d_init=d_init,
#                    n_init=n_init),
#         layer.BatchPowerNorm(mode=mode),
#         layer.MIMOFOEAf(name='FOEAf',
#                         w0=w0,
#                         train=mimo_train,
#                         preslicer=core.conv1d_slicer(rtaps),
#                         foekwargs={}),
#         layer.vmap(layer.Conv1d)(name='RConv', taps=rtaps),  # vectorize column-wise Conv1D
#         layer.MIMOAF(train=mimo_train))  # adaptive MIMO layer
        
#     return base

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
        layer.MIMOFOEAf1(name='FOEAf',
                        w0=w0,
                        train=mimo_train,
                        preslicer=core.conv1d_slicer(rtaps),
                        foekwargs={}),
        layer.vmap(layer.Conv1d)(name='RConv', taps=rtaps),
        layer.MIMOAF1(train=mimo_train),
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

#     # (A) 我们只定义一个“融合后”的 FDBP 分支 (包含 fdbp + fdbp1 + bridge)
#     fused_branch = layer.Serial(
#         layer.FDBP(steps=steps, 
#                             dtaps=dtaps,
#                             ntaps=ntaps,
#                             ixpm_window=7,
#                             d_init=d_init,
#                             n_init=n_init,
#                             name='fdbp2branches'),   # 见下文如何包装
#         layer.BatchPowerNorm(mode=mode),
#         layer.MIMOFOEAf(name='FOEAf',
#                         w0=w0,
#                         train=mimo_train,
#                         preslicer=core.conv1d_slicer(rtaps),
#                         foekwargs={}),
#         layer.vmap(layer.Conv1d)(name='RConv', taps=rtaps),
#         layer.MIMOAF(train=mimo_train),
#         name='fused_all_in_one'
#     )

#     # (B) 直接返回这条单分支即可
#     #     若仍想和别的分支再并行，也可以，但这里示例是纯单分支。
#     return fused_branch


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
        
def si_snr_flattened(
    target: jnp.ndarray, 
    estimate: jnp.ndarray,
    eps: float = 1e-8
) -> jnp.ndarray:
    """
    将( T, 2 )形状(双偏振)的时域信号展开成( 2T, )后,
    计算单通道意义下的SI-SNR, 返回其负值做loss.
    """

    # 假设 (T,2)
    # flatten => (2T,)
    target_1d = jnp.reshape(target, (-1,))
    estimate_1d = jnp.reshape(estimate, (-1,))

    # 然后调用和单通道同样的si_snr逻辑
    dot_product = jnp.sum(target_1d * estimate_1d)
    target_energy = jnp.sum(target_1d**2) + eps
    s_target = (dot_product / target_energy) * target_1d

    e_noise = estimate_1d - s_target
    t_energy = jnp.sum(s_target**2)
    n_energy = jnp.sum(e_noise**2)

    si_snr_value = 10.0 * jnp.log10((t_energy + eps) / (n_energy + eps))
    return -si_snr_value
        
import jax.numpy as jnp

import jax
import jax.numpy as jnp

def unsupervised_snr_jones(
    x: jnp.ndarray,  # shape (T, 2), complex dtype
    v: jnp.ndarray,  # shape (2,), complex dtype (trainable or from your model)
    eps: float = 1e-8
) -> jnp.ndarray:
    """
    利用双偏振观测的 Rank-1 近似,
    构造无监督的 'SNR' 风格损失:
      1) 给定 x(t) in C^2,
      2) 令 s(t) = <x(t), v^*> / <v, v^*>, 
      3) 令 x_approx(t) = s(t) * v,
      4) 噪声 e(t) = x(t) - x_approx(t),
      5) 以 sum(|x_approx|^2) / sum(|e|^2) 做 SNR 度量, 并取负值.
    
    参数:
      x : (T, 2) 复数张量, 表示T个时刻的双偏振信号
      v : (2,)   复数向量, 对应 "Jones向量" 或 主偏振方向
      eps: 浮点安全项
    
    注意:
      - 为了让此函数可微, 需要 x, v 都是 jnp.array;
      - 如果要训练 v, 需在外部用 jax.grad / jax.jit 包起来一起优化;
      - s(t) 是根据最小二乘投影显式计算, 也会对 v 产生梯度.
    """

    # (1) 计算 v 的范数^2
    v_norm2 = jnp.sum(jnp.abs(v)**2) + eps

    # (2) 对每个 t 计算 s(t) 
    # s(t) = ( x(t) 与 v 的内积 ) / ( ||v||^2 )
    # x(t)* conj(v) => 这里把 x 视为行向量, v 视为列向量
    # shape (T,)
    st = jnp.sum(x * jnp.conj(v), axis=-1) / v_norm2

    # (3) 重建 x_approx(t)
    #  x_approx(t) = s(t) * v, shape => (T, 2)
    # 需要 broadcasting
    x_approx = st[:, None] * v[None, :]

    # (4) 残差 e
    e = x - x_approx

    # (5) 计算信号能量与噪声能量
    signal_energy = jnp.sum(jnp.abs(x_approx)**2)
    noise_energy  = jnp.sum(jnp.abs(e)**2)

    snr_value = 10.0 * jnp.log10( (signal_energy + eps) / (noise_energy + eps) )

    # 训练时取负值做loss
    return -snr_value

     
        
# def loss_fn(module: layer.Layer,
#             params: Dict,
#             state: Dict,
#             y: Array,
#             x: Array,
#             aux: Dict,
#             const: Dict,
#             sparams: Dict,):
#     params = util.dict_merge(params, sparams)
#     z_original, updated_state = module.apply(
#         {'params': params, 'aux_inputs': aux, 'const': const, **state}, core.Signal(y)) 
#     # y_transformed = apply_combined_transform(y)
#     aligned_x = x[z_original.t.start:z_original.t.stop]
#     mse_loss = jnp.mean(jnp.abs(z_original.val - aligned_x) ** 2)
#     snr = si_snr(jnp.abs(z_original.val), jnp.abs(aligned_x)) 
#     # snr = si_snr_flattened(jnp.abs(z_original.val), jnp.abs(aligned_x)) 
#     return snr, updated_state

        
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
    aligned_x = x[z_original.t.start:z_original.t.stop]
    # mse_loss = jnp.mean(jnp.abs(z_original.val - aligned_x) ** 2)
    snr = unsupervised_snr_svd(jnp.abs(z_original.val)) 
    return snr, updated_state

              
############# GMI-LOSS WITH TWO PART TRINING ###################
# import jax
# import jax.numpy as jnp

                                
# def get_16qam_constellation() -> jnp.ndarray:
#     """
#     定义16QAM的星座点，返回形状为 (16, 2) 的实数数组，
#     每个星座点以 [real, imag] 表示。
#     取未归一化的 [-3, -1, 1, 3] 网格，然后归一化到单位平均能量。
#     """
#     re = jnp.array([-3, -1, 1, 3], dtype=jnp.float32)
#     im = jnp.array([-3, -1, 1, 3], dtype=jnp.float32)
#     points = jnp.array([[r, i] for r in re for i in im])
#     norm_factor = jnp.sqrt(jnp.mean(jnp.sum(points**2, axis=-1)))
#     return points / norm_factor

# def to_real(x: jnp.ndarray) -> jnp.ndarray:
#     """
#     将输入 x 转换为实数表示，输出形状为 (batch, 2)。
#     规则如下：
#       - 如果 x 为复数数组，则转换为 [real, imag] 表示；
#       - 如果 x 已为实数，但最后一维不是2，则将除 batch 和最后一维外的所有维度 flatten，
#         然后对每个样本取平均，得到形状 (batch,2)。
#     """
#     if jnp.iscomplexobj(x):
#         x = jnp.stack([jnp.real(x), jnp.imag(x)], axis=-1)
#     # 假设 x 形状为 (batch, ..., 2)
#     batch = x.shape[0]
#     # 如果中间维度的乘积大于 1，则 reshape 为 (batch, extra, 2) 后取平均
#     if x.ndim > 2:
#         extra = 1
#         for d in x.shape[1:-1]:
#             extra *= d
#         if extra > 1:
#             x = jnp.reshape(x, (batch, -1, 2))
#             x = jnp.mean(x, axis=1)
#         else:
#             x = jnp.reshape(x, (batch, 2))
#     else:
#         # 如果 x 维度已经是 (batch,2) 则直接返回
#         x = jnp.reshape(x, (batch, 2))
#     if x.shape[-1] != 2:
#         raise ValueError(f"Expected last dimension to be 2 for [real, imag] representation, got shape {x.shape}")
#     return x

# def gmi_loss_16qam(
#     target: jnp.ndarray,
#     estimate: jnp.ndarray,
#     constellation: jnp.ndarray = None,
#     eps: float = 1e-8,
# ) -> jnp.ndarray:
#     """
#     针对16QAM的GMI计算，并返回“负”的平均GMI，方便当作loss使用。
#     """
#     if constellation is None:
#         constellation = get_16qam_constellation()  # shape (16, 2)

#     # 转 (batch, 2) 实数
#     target = to_real(target)     # (batch, 2)
#     estimate = to_real(estimate) # (batch, 2)

#     # 估计平均噪声功率 sigma^2
#     noise = estimate - target  # shape (batch,2)
#     sigma2 = jnp.mean(jnp.sum(noise**2, axis=-1)) + eps  # 一个标量

#     # 计算与星座点的欧氏距离及似然
#     # diff: (batch, 16, 2)
#     diff = estimate[:, None, :] - constellation[None, :, :]
#     dist_sq = jnp.sum(diff**2, axis=-1)           # (batch, 16)
#     likelihoods = jnp.exp(-dist_sq / sigma2)      # (batch, 16)

#     # 分母：对所有星座点似然做加权（先验1/16），可以写成 sum(...) / 16
#     denom = jnp.sum(likelihoods, axis=-1) + eps   # (batch,)

#     # 找到 target 对应（或最近）的那一个星座点
#     def get_index(t):
#         distances = jnp.sum((constellation - t)**2, axis=-1)
#         return jnp.argmin(distances)
#     indices = jax.vmap(get_index)(target)  # (batch,)

#     # 取“正确”点的似然
#     likelihood_true = likelihoods[jnp.arange(target.shape[0]), indices]  # (batch,)

#     # 对数似然比: log2( (1/16)*likelihood_true / (1/16)*sum(likelihoods) )
#     # 省略 1/16 后，只差常数项，对优化无影响
#     ratio = (likelihood_true + eps) / (denom / 16.0 + eps)
#     # 注意 jnp.log2(x) 可以写成 jnp.log(x) / jnp.log(2)
#     log_ratio = jnp.log(ratio) / jnp.log(2.0)

#     # GMI = 平均 (batch 维度)，最后取负号做 loss
#     gmi = jnp.mean(log_ratio)
#     return gmi
        
# def loss_fn(module: layer.Layer,
#             params: Dict,
#             state: Dict,
#             y: Any,
#             x: Any,
#             aux: Dict,
#             const: Dict,
#             sparams: Dict,
#             loss_type: str = 'si_snr'):
#     """
#     扩展后的 loss_fn 支持三种损失计算方式：
#       - 'si_snr'  : SI-SNR loss（适用于复数信号，使用共轭内积）
#       - 'gmi_loss': 针对16QAM的 GMI loss（基于高斯似然计算互信息）
#       - 'combined': SI-SNR loss + 0.1 * GMI loss

#     注意：假设网络输出和目标信号均为复数信号（或 [real, imag] 格式）。
#     """
#     # 合并参数
#     params = util.dict_merge(params, sparams)
#     # 应用模块，得到输出信号及更新后的状态
#     z_original, updated_state = module.apply(
#         {'params': params, 'aux_inputs': aux, 'const': const, **state}, core.Signal(y))
#     # 对齐目标信号
#     aligned_x = x[z_original.t.start:z_original.t.stop]
    
#     if loss_type == 'si_snr':
#         loss = si_snr(jnp.abs(z_original.val), jnp.abs(aligned_x)) 
#     elif loss_type == 'gmi_loss':
#         loss = gmi_loss_16qam(z_original.val, aligned_x)
#     elif loss_type == 'combined':
#         loss = si_snr(jnp.abs(z_original.val), jnp.abs(aligned_x))  + 0.1 * gmi_loss_16qam(z_original.val, aligned_x)
#     else:
#         raise ValueError("Unknown loss type: " + loss_type)
    
#     return loss, updated_state

# @partial(jit, backend='cpu', static_argnums=(0, 1, 10))
# def update_step_with_loss_type(
#     module: layer.Layer,       # arg0
#     opt: cxopt.Optimizer,      # arg1
#     i: int,                    # arg2
#     opt_state: tuple,          # arg3
#     module_state: Dict,        # arg4
#     y: Array,                  # arg5
#     x: Array,                  # arg6
#     aux: Dict,                 # arg7
#     const: Dict,               # arg8
#     sparams: Dict,             # arg9
#     loss_type: str = 'si_snr'  # arg10 (被指定成 static_argnums=(...,10))
# ):
#     """
#     函数签名中多一个 loss_type，用来调用不同的损失函数分支。
#     """
#     # 1) 拿到可训练参数
#     params = opt.params_fn(opt_state)

#     # 2) 我们包一层函数，这样就能把 loss_type 传入 loss_fn
#     def wrapped_loss_fn(params_, module_state_):
#         return loss_fn(module, params_, module_state_, y, x,
#                        aux, const, sparams, loss_type=loss_type)

#     # 3) 计算loss + grads
#     (loss, module_state), grads = value_and_grad(
#         wrapped_loss_fn, argnums=0, has_aux=True
#     )(params, module_state)

#     # 4) 更新 opt_state
#     opt_state = opt.update_fn(i, grads, opt_state)
#     return loss, opt_state, module_state

# def train(model: Model,
#           data: gdat.Input,
#           batch_size: int = 500,
#           n_iter: int = 3000,  # 总共2000步
#           opt: optim.Optimizer = optim.adam(optim.piecewise_constant(
#               [500, 1000], [1e-4, 1e-5, 1e-6]))
#          ):
#     """
#     单阶段写法，但在循环内部通过 if/else 分段切换损失。
#     """
#     params, module_state, aux, const, sparams = model.initvar
#     opt_state = opt.init_fn(params)

#     n_batch, batch_gen = get_train_batch(data, batch_size, model.overlaps)
#     n_iter = min(n_iter, n_batch)  # 防止数据不够

#     # 把batch数据一次性取出来（可选）
#     batch_gen = list(batch_gen)

#     for i in tqdm(range(n_iter), desc='training'):
#         # 1) 取第 i 个 batch
#         y, x = batch_gen[i]

#         # 2) 切换loss
#         if i < 2500:
#             loss_type = 'gmi_loss'
#         else:
#             loss_type = 'gmi_loss'
        
#         # 3) 用 update_step_with_loss_type 或类似的函数
#         #    （假设它能按loss_type调用不同的损失）
#         aux = core.dict_replace(aux, {'truth': x})
#         loss, opt_state, module_state = update_step_with_loss_type(
#             module = model.module,
#             opt = opt,
#             i = i,
#             opt_state = opt_state,
#             module_state = module_state,
#             y = y,
#             x = x,
#             aux = aux,
#             const = const,
#             sparams = sparams,
#             loss_type = loss_type
#         )

#         # 4) yield or return or store
#         yield (loss, opt.params_fn(opt_state), module_state)       
############# GMI-LOSS WITH TWO PART TRINING ###################

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

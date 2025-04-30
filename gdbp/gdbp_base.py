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

import jax, jax.numpy as jnp, jax.lax as lax

def evm_ring_focal(tx, rx,
                   tau=0.05, gamma=2.0,
                   thr_in=None, thr_mid=None,
                   w_in=1.0, w_mid=1.3, w_out=1.8,
                   eps=1e-8,
                   step=None):                 # ← 传 step 进来
    r    = jnp.abs(tx)
    err2 = jnp.abs(rx - tx)**2
    sig2 = jnp.abs(tx)**2

    # —— 自动阈值 ————————————————————
    r_max = jnp.max(r)
    thr_in  = 0.33*r_max if thr_in  is None else thr_in
    thr_mid = 0.66*r_max if thr_mid is None else thr_mid

    # —— 仅在 step==0 时打印一次 r 统计 —————
    def _print(_):
        jax.debug.print("r min/mean/max = {0:.3f}/{1:.3f}/{2:.3f}",
                        jnp.min(r), jnp.mean(r), r_max)
        return 0
    lax.cond(step==0, _print, lambda _:0, operand=None)

    # —— focal 权重 & masks ————————————
    focal = (err2 / tau)**gamma
    m_in  = (r <  thr_in ).astype(jnp.float32)
    m_mid = ((r>=thr_in) & (r < thr_mid)).astype(jnp.float32)
    m_out = (r >= thr_mid).astype(jnp.float32)

    def _evm(mask):
        num = jnp.sum(err2 * focal * mask)
        den = jnp.sum(sig2 * mask) + eps
        return num / den

    return (w_in  * _evm(m_in)  +
            w_mid * _evm(m_mid) +
            w_out * _evm(m_out))


def phase_err(tx, rx):
    return jnp.mean(jnp.angle(rx) - jnp.angle(tx))**2
        
def si_snr_flat_amp_pair(tx, rx, eps=1e-8):
    """幅度|·|，两极化展平后算 α，再平分给两路"""
    s  = jnp.reshape(jnp.abs(tx), (-1,))
    x  = jnp.reshape(jnp.abs(rx), (-1,))
    alpha = jnp.vdot(s, x).real / (jnp.vdot(s, s).real + eps)
    e  = x - alpha * s
    snr_db = 10. * jnp.log10( (jnp.vdot(alpha*s, alpha*s).real + eps) /
                              (jnp.vdot(e, e).real + eps) )
    return -snr_db                   
        


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
#     snr = si_snr_flat_amp_pair(jnp.abs(z_original.val), jnp.abs(aligned_x)) 
#     evm = evm_ring(jnp.abs(z_original.val), jnp.abs(aligned_x)) 
#     # snr = snr + 0.01 * evm
#     return snr, updated_state

def loss_fn(module, params, state,
            y, x, aux, const, sparams,
            step):                   # ← 新参数

    p_all = util.dict_merge(params, sparams)
    z, new_state = module.apply(
        {'params': p_all, 'aux_inputs': aux,
         'const': const, **state},
        core.Signal(y))

    tx = x[z.t.start:z.t.stop]
    rx = z.val

    # —— α 归一化 ————————————————
    alpha = jnp.vdot(jnp.abs(tx), jnp.abs(rx)).real / \
            jnp.vdot(jnp.abs(tx), jnp.abs(tx)).real
    rx_n  = rx / alpha

    # —— SNR & EVM ————————————————
    snr_loss = si_snr_flat_amp_pair(jnp.abs(tx), jnp.abs(rx_n))
    evm_loss = evm_ring_focal(jnp.abs(tx), jnp.abs(rx_n),
                              tau=0.05, gamma=2.0,
                              step=step)          # ★ 传 step

    total = snr_loss + 0.008 * evm_loss
    return total, new_state


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
#     return snr, updated_state


#### LMS - LOSS #####
# def loss_fn(module: layer.Layer,
#             params: Dict, state: Dict,
#             y: Array, x: Array,
#             aux: Dict, const: Dict, sparams: Dict
#            ) -> Tuple[jnp.ndarray, Dict]:
#     params = util.dict_merge(params, sparams)

#     # a. RMS‑norm（match sent symbols）
#     r = _rms(x)
#     y, x = y / r, x / r

#     # b. 前向
#     z, new_state = module.apply(
#         {'params': params, 'aux_inputs': aux,
#          'const': const, **state},
#         core.Signal(y))

#     x_aligned = x[z.t.start:z.t.stop]

#     # c. 投影‑MSE
#     # loss = _proj_mse(z.val, x_aligned)
#     loss = si_snr(jnp.abs(z.val), jnp.abs(x_aligned)) 
#     return loss, new_state    


# @partial(jit, backend='cpu', static_argnums=(0, 1))
# def update_step(module: layer.Layer,
#                 opt: cxopt.Optimizer,
#                 i: int,
#                 opt_state: tuple,
#                 module_state: Dict,
#                 y: Array,
#                 x: Array,
#                 aux: Dict,
#                 const: Dict,
#                 sparams: Dict):
#     ''' single backprop step

#         Args:
#             model: model returned by `model_init`
#             opt: optimizer
#             i: iteration counter
#             opt_state: optimizer state
#             module_state: module state
#             y: transmitted waveforms
#             x: aligned sent symbols
#             aux: auxiliary input
#             const: contants (internal info generated by model)
#             sparams: static parameters

#         Return:
#             loss, updated module state
#     '''

#     params = opt.params_fn(opt_state)
#     (loss, module_state), grads = value_and_grad(
#         loss_fn, argnums=1, has_aux=True)(module, params, module_state, y, x,
#                                           aux, const, sparams)
#     opt_state = opt.update_fn(i, grads, opt_state)
#     return loss, opt_state, module_state

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
    """
    One backward & optimizer step
    ---------------------------------
    • module  : Layer  (Flax style)
    • opt     : cxopt.Optimizer
    • i       : iteration counter  (传给 loss_fn 作 debug / 阶段调度)
    """

    # 1) 当前参数
    params = opt.params_fn(opt_state)

    # 2) 计算损失 & grads   —— 传 step=i --------------------
    (loss, new_state), grads = value_and_grad(
        loss_fn,               # 已改签名：最后一个位置参数是 step
        argnums=1,             # grads w.r.t. params
        has_aux=True)(
            module, params, module_state,
            y, x, aux, const, sparams,
            i)                 # ★ 传递迭代号 ★

    # 3) 优化器一步
    opt_state   = opt.update_fn(i, grads, opt_state)

    # 4) 把新的 module_state 返还
    return loss, opt_state, new_state
                  
                        
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
                                     
def train_once(model_tr, data_tr, 
               batch_size=500, n_iter=3000):
    """return  params , state_bundle"""
    params, module_state, aux, const, sparams = model_tr.initvar

    # ——优化器
    opt = optim.adam(
            optim.piecewise_constant([500,1000],[1e-4,1e-5,1e-6]))
    opt_state = opt.init_fn(params)

    # ——batch 生成
    n_batch, batch_gen = get_train_batch(
        data_tr, batch_size, model_tr.overlaps)

    for i,(y,x) in tqdm(enumerate(batch_gen),
                        total=min(n_iter,n_batch), desc='train', leave=False):
        if i >= n_iter: break
        aux = core.dict_replace(aux, {'truth': x})   # 保存真符号
        loss, opt_state, module_state = update_step(
            model_tr.module, opt, i, opt_state,
            module_state, y, x, aux, const, sparams)

    params = opt.params_fn(opt_state)
    state_bundle = (module_state, aux, const, sparams)
    return params, state_bundle        
                       
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
                 
def test_once(model: Model, params: Dict, state_bundle, data: gdat.Input,
              eval_range=(300_000, -20_000), metric_fn=comm.qamqot):
    module_state, aux, const, sparams = state_bundle
    aux = core.dict_replace(aux, {'truth': data.x})

    z,_ = jax.jit(model.module.apply, backend='cpu')(
        {'params': util.dict_merge(params, sparams),
         'aux_inputs': aux, 'const': const, **module_state},
        core.Signal(data.y))
                      
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

                
##### LMS - LOSS #####
# def test(model: Model,
#          params: Dict,
#          data: gdat.Input,
#          eval_range: tuple=(300000, -20000),
#          metric_fn=comm.qamqot):

#     state, aux, const, sparams = model.initvar[1:]
#     aux = core.dict_replace(aux, {'truth': data.x})
#     if params is None:
#       params = model.initvar[0]
            
#     r = np.sqrt(np.mean(np.abs(data.x)**2))
#     y_norm = data.y / r
#     x_norm = data.x / r
#     z, _ = jit(model.module.apply,
#                backend='cpu')({
#                    'params': util.dict_merge(params, sparams),
#                    'aux_inputs': aux,
#                    'const': const,
#                    **state
#                }, core.Signal(y_norm))
#     metric = metric_fn(z.val,
#                        x_norm[z.t.start:z.t.stop],
#                        scale=np.sqrt(10),
#                        eval_range=eval_range)
#     return metric, z

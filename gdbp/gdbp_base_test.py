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
#         layer.BatchPowerNorm(mode=mode),
#         layer.MIMOFOEAf(name='FOEAf',
#                         w0=w0,
#                         train=mimo_train,
#                         preslicer=core.conv1d_slicer(rtaps),
#                         foekwargs={}),
#         layer.vmap(layer.Conv1d)(name='RConv', taps=rtaps),
#         layer.MIMOAF(train=mimo_train),
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
# ── ★ 工具 1：估算还能涨多少 Q(dB) ───────────────────────────
def q_gain_upper(tx, rx):
    """
    tx, rx: shape (N,2) complex
    返回当前链路把结构性误差全部抹掉时，理论还能涨多少 Q(dB)
    """
    e      = rx - tx
    r      = jnp.abs(tx)
    E_tot  = jnp.mean(jnp.abs(e)**2)
    # 16‑QAM 下外环(r>=1.1) 基本≈剩余非线性/ISI
    ΔE     = jnp.mean(jnp.abs(e)[r >= 1.1]**2)
    return 10 * jnp.log10(1.0 + ΔE / E_tot) 

def _collect_watch_kernels(tree, path=()):
    """
    深度遍历 param/grad 树，返回所有满足
    “路径中含 WATCH_CONV_KEYS 中任一关键字且有 'kernel' ” 的节点
    """
    out = []
    for k, v in tree.items():
        new_path = path + (k,)
        if isinstance(v, (dict, flax.core.FrozenDict)):
            out += _collect_watch_kernels(v, new_path)
        elif k == 'kernel':
            joined = '/'.join(new_path)        # e.g. fdbp_series/RConv1/kernel
            if any(key in joined for key in WATCH_CONV_KEYS):
                out.append(v)                  # v 是 ndarray
    return out

def _snr_loss(tx, rx, eps=1e-8):
    """跟主 loss 同式的 SNR-loss，用于反向求梯度"""
    s = jnp.reshape(jnp.abs(tx), (-1,))
    r = jnp.reshape(jnp.abs(rx), (-1,))
    α = jnp.vdot(s, r).real / (jnp.vdot(s, s).real + eps)
    e = r - α * s
    return -10. * jnp.log10((jnp.vdot(α * s, α * s).real + eps) /
                            (jnp.vdot(e, e).real + eps))

def tap_grad_snapshot(module, params, state, const,
                      aux, y, x, *, watch_keys=('DConv',),
                      split=False, debug=True):
    """
    返回选中 kernel 的逐-tap L2 梯度
    split=False → np.ndarray 1-D
    split=True  → {path: ndarray}
    """
    def _loss(p):
        z, _ = module.apply({'params': p, **state,
                             'const': const,
                             'aux_inputs': aux}, core.Signal(y))
        tx = x[z.t.start:z.t.stop]
        return _snr_loss(tx, z.val)

    grads = jax.grad(_loss)(params)

    # ---- 递归抓取 ----------------------------------------------------------
    bucket = {}
    def _rec(tree, path=()):
        for k, v in tree.items():
            newp = path + (k,)
            if isinstance(v, (dict, flax.core.FrozenDict)):
                _rec(v, newp)
            elif k == 'kernel' and any(kwd in '/'.join(newp) for kwd in watch_keys):
                bucket['/'.join(newp[:-1])] = v
    _rec(grads)

    if debug:
        print('---- watched kernels ----')
        for p, g in bucket.items():
            print(f'{p:<45s}  shape={tuple(g.shape)}')
        print('total tap bins =', sum(g.shape[0] for g in bucket.values()))

    def _l2(g):
        g2 = g.real ** 2 + g.imag ** 2
        return jnp.sqrt(jnp.sum(g2, axis=tuple(range(1, g.ndim))))

    if split:
        return {k: np.asarray(_l2(v)) for k, v in bucket.items()}
    return np.concatenate([np.asarray(_l2(v)) for v in bucket.values()])


def plot_impulse_response(model: Model, params_bundle, taps=4096):
    params, state, aux, const, _ = params_bundle

    # --- 改这里：做 2-pol 复数冲激 -------------------------
    imp = np.zeros((taps, 2), dtype=np.complex64)
    imp[taps // 2, :] = 1.0 + 0j                   # 中心 Dirac，双极化同振幅
    # -----------------------------------------------------

    z, _ = model.module.apply({'params': params, **state,
                               'const': const, 'aux_inputs': aux},
                              core.Signal(imp))
    plt.figure(figsize=(6,2))
    plt.plot(np.abs(np.asarray(z.val[:, 0])), label='pol-0')
    plt.plot(np.abs(np.asarray(z.val[:, 1])), '--', label='pol-1')
    plt.title('|h[n]| after D-Conv'); plt.legend(); plt.tight_layout(); plt.show()


def show_grad_stat(grad_log):
    """画 Var / Mean 随迭代的变化"""
    var = [np.var(g) for _, g in grad_log]
    mean = [np.mean(g) for _, g in grad_log]
    iters = [step for step, _ in grad_log]

    plt.figure(); plt.semilogy(iters, var, '.-')
    plt.title('Var(||∇w||) vs iter'); plt.xlabel('iter'); plt.grid(); plt.show()

    plt.figure(); plt.semilogy(iters, mean, '.-')
    plt.title('Mean(||∇w||) vs iter'); plt.xlabel('iter'); plt.grid(); plt.show()

def gamma_sweep(model: Model,
                base_params_bundle,
                ds_val: gdat.Input,
                scales=(0.5, 1.0, 1.5)):
    """
    把 N-Conv (= γ) 整体乘以 scale，测试 Q-factor 灵敏度
    """
    params, state, aux, const, sparams = base_params_bundle
    base = util.dict_merge(params, sparams)

    def _scale_gamma(tree, k):
        # N-Conv kernel 形状固定为 (1,2,2)
        return util.tree_map(lambda v:
                             v * k if v.shape == (1, 2, 2) else v,
                             tree)

    for k in scales:
        new_p = _scale_gamma(base, k)            # 更新 γ
        metric, _ = test(model, new_p, ds_val)   # ← 直接接住两个返回值
        print(f"γ × {k:<4}:  Q = {metric.QSq.total:5.3f} dB")




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
    snr = si_snr_flat_amp_pair(jnp.abs(z_original.val), jnp.abs(aligned_x)) 
    evm = evm_ring(jnp.abs(z_original.val), jnp.abs(aligned_x)) 
    snr = snr + 0.01 * evm
    return snr, updated_state

# def loss_fn(module: layer.Layer,
#             params: Dict, state: Dict,
#             y: Array, x_sym: Array,
#             aux: Dict, const: Dict, sparams: Dict,
#             eps=1e-8):
#     # ---- 合并参数 & 前向 ------------------------------------------
#     p_all = util.dict_merge(params, sparams)
#     z_sig, new_state = module.apply(
#         {'params': p_all, 'aux_inputs': aux,
#          'const': const, **state},
#         core.Signal(y))

#     # ---- 发送符号重复 sps=2 → 与波形同采样 --------------------------
#     sps   = z_sig.t.sps                        # =2
#     x_rep = jnp.repeat(x_sym, sps, axis=0)     # (N_sym*sps, 2)

#     # ---- 对齐 & 截短 ---------------------------------------------
#     x_ref = x_rep[z_sig.t.start : z_sig.t.stop]
#     L     = min(z_sig.val.shape[0], x_ref.shape[0])
#     z_val = z_sig.val[:L]
#     x_val = x_ref  [:L]

#     # ---- SNR-amp + 0.01·EVM_ring ---------------------------------
#     snr = si_snr_flat_amp_pair(jnp.abs(z_val), jnp.abs(x_val), eps=eps)
#     evm = evm_ring(jnp.abs(z_val), jnp.abs(x_val), eps=eps)
#     loss = snr + 0.01 * evm
#     return loss, new_state


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
                               
# def train(model: Model,
#           data: gdat.Input,
#           batch_size: int = 500,
#           n_iter: int | None = None,
#           opt: optim.Optimizer = optim.adam(
#               optim.piecewise_constant([500, 1000], [1e-4, 1e-5, 1e-6])),
#           *,                    # 下面两个是新增调试开关，可省略
#           debug_first_iter=True,
#           debug_every=0):
#     """
#     1-epoch 训练生成器  
#     每迭代返回 (loss, params, module_state)

#     ⚙️ 额外调试：
#       • debug_first_iter=True  ➜ 第 0 个 batch 打印一次齐头对齐检查
#       • debug_every=N          ➜ 每 N 步再打印一次（0 表示只首批）
#     """
#     params, mod_state, aux, const, sparams = model.initvar
#     opt_state = opt.init_fn(params)

#     n_batch, batch_gen = get_train_batch(data, batch_size, model.overlaps)
#     n_iter = n_batch if n_iter is None else min(n_iter, n_batch)

#     for i, (y, x) in tqdm(enumerate(batch_gen),
#                           total=n_iter, desc='train', leave=False):
#         if i >= n_iter:
#             break

#         # 把真符号塞进 aux 供模块内部自适应 (FOE / MIMO 等) 用
#         aux = core.dict_replace(aux, {'truth': x})

#         # ---- ① 反向传播一步 ------------------------------------------------
#         loss, opt_state, mod_state = update_step(
#             model.module, opt, i, opt_state,
#             mod_state, y, x, aux, const, sparams)

#         # ---- ② 可选调试：对齐长度 / NaN 检查 -------------------------------
#         if (debug_first_iter and i == 0) or (debug_every and i % debug_every == 0):
#             with jax.disable_jit():                      # 避免重复编译
#                 p_all = util.dict_merge(opt.params_fn(opt_state), sparams)
#                 z_dbg, _ = model.module.apply(
#                     {'params': p_all,
#                      'aux_inputs': aux,
#                      'const': const,
#                      **mod_state},
#                     core.Signal(y))
#                 x_dbg = x[z_dbg.t.start: z_dbg.t.stop]
#                 L = min(z_dbg.val.shape[0], x_dbg.shape[0])
#                 print(f"[DEBUG] step={i:4d}  z_len={z_dbg.val.shape[0]}  "
#                       f"x_len={x_dbg.shape[0]}  L={L}  "
#                       f"finite(z)={jnp.isfinite(z_dbg.val).all()}  "
#                       f"finite(x)={jnp.isfinite(x_dbg).all()}")
#         # --------------------------------------------------------------------

#         yield loss, opt.params_fn(opt_state), mod_state



# def train(model: Model, data: gdat.Input,
#           batch_size: int = 500,
#           n_iter: int | None = None,
#           opt: optim.Optimizer = optim.adam(
#                 optim.piecewise_constant([500, 1000], [1e-4, 1e-5, 1e-6])),
#           *,                       # 新增可选项
#           grad_keys=('RConv1',),     # 想监控哪些卷积
#           split_grad=False):        # True → 每 kernel 单独画
#     """
#     直接覆盖原 train()，其余代码 *不需要* 改。
#     """
#     params, state, aux, const, sparams = model.initvar
#     opt_state = opt.init_fn(params)

#     n_batch, batch_gen = get_train_batch(data, batch_size, model.overlaps)
#     n_iter = n_batch if n_iter is None else min(n_iter, n_batch)

#     grad_log = []          # ← 收集梯度向量做统计
#     for i, (y, x) in tqdm(enumerate(batch_gen), total=n_iter,
#                            desc='train', leave=False):
#         if i >= n_iter: break
#         aux = core.dict_replace(aux, {'truth': x})

#         # --- 更新 ----------------------------------------------------------------
#         loss, opt_state, state = update_step(
#             model.module, opt, i, opt_state,
#             state, y, x, aux, const, sparams)

#         # --- Q-gain 上限 ----------------------------------------------------------
#         if i % 500 == 0:
#             p_all = util.dict_merge(opt.params_fn(opt_state), sparams)
#             z, _ = model.module.apply({'params': p_all, **state,
#                                        'const': const, 'aux_inputs': aux},
#                                       core.Signal(y))
#             gain = q_gain_upper(x[z.t.start: z.t.stop], z.val)
#             print(f"[{i:4d}]  当前 Q 上限 ≈ +{gain:.3f} dB")

#         # --- 梯度可视化 -----------------------------------------------------------
#         if i % 1000 == 0:
#             g_out = tap_grad_snapshot(
#                 model.module,
#                 util.dict_merge(opt.params_fn(opt_state), sparams),
#                 state, const, aux, y, x,
#                 watch_keys=grad_keys,
#                 split=split_grad, debug=not split_grad)

#             if split_grad:
#                 for name, vec in g_out.items():
#                     plt.figure(figsize=(6, 1.6))
#                     plt.bar(range(len(vec)), vec)
#                     plt.yscale('log'); plt.title(f'iter {i}  {name}')
#                     plt.tight_layout(); plt.show()
#                 grad_log.append((i, np.concatenate(list(g_out.values()))))
#             else:
#                 plt.figure(figsize=(6, 2))
#                 plt.bar(range(len(g_out)), g_out)
#                 plt.yscale('log'); plt.title(f'iter {i}  per-tap ∥∇w∥')
#                 plt.tight_layout(); plt.show()
#                 grad_log.append((i, g_out))

#         yield loss, opt.params_fn(opt_state), state
#     if split_grad and bucket:                      # split=True 时
#         last_g = np.concatenate(list(bucket.values()))
#     else:
#         last_g = g_out if 'g_out' in locals() else None
#     grad_log.append((i, last_g, opt.params_fn(opt_state)))
#     return grad_log   # ← 方便后处理


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

# def test(model: Model,
#          params: Dict | None,
#          data: gdat.Input,
#          eval_range=(300_000, -20_000),
#          metric_fn = comm.qamqot,
#          verbose=False):

#     state, aux, const, sparams = model.initvar[1:]
#     aux = core.dict_replace(aux, {'truth': data.x})
#     if params is None:
#         params = model.initvar[0]

#     # -------- forward ------------------------------------------------
#     z_sig, _ = jax.jit(model.module.apply, backend='cpu')(
#         {'params': util.dict_merge(params, sparams),
#          'aux_inputs': aux, 'const': const, **state},
#         core.Signal(data.y))

#     # -------- repeat sent-symbols -----------------------------------
#     sps   = z_sig.t.sps
#     x_rep = jnp.repeat(data.x, sps, axis=0)       # 波形采样率
#     x_ref = x_rep[z_sig.t.start : z_sig.t.stop]

#     L     = min(z_sig.val.shape[0], x_ref.shape[0])
#     z_aln, x_aln = z_sig.val[:L], x_ref[:L]

#     if L <= eval_range[0]:
#         eval_range = (0, 0)

#     if verbose:
#         print(f"[TEST] z_len={z_sig.val.shape[0]}  x_len={x_ref.shape[0]} "
#               f"L={L} finite(z)={jnp.isfinite(z_aln).all()} "
#               f"finite(x)={jnp.isfinite(x_aln).all()}")

#     metric = metric_fn(z_aln, x_aln, scale=np.sqrt(10),
#                        eval_range=eval_range)
#     return metric, z_sig


                 
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

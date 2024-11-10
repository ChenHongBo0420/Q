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
        layer.BatchPowerNorm1(mode=mode),
        layer.MIMOFOEAf1(name='FOEAf',
                        w0=w0,
                        train=mimo_train,
                        preslicer=core.conv1d_slicer(rtaps),
                        foekwargs={}),
        layer.vmap(layer.Conv1d1)(name='RConv', taps=rtaps),
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
        layer.FanInConcat()
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
#     # mse_loss = jnp.mean(jnp.abs(z_original.val - aligned_x) ** 2)
#     snr = si_snr(jnp.abs(z_original.val), jnp.abs(aligned_x)) 
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
    concatenated_output, updated_state = module.apply(
        {'params': params, 'aux_inputs': aux, 'const': const, **state}, core.Signal(y))

    # 获取与模型输出对齐的真实信号
    aligned_x = x[concatenated_output.t.start:concatenated_output.t.stop]
    output_dbp, output_nn = jnp.split(concatenated_output.val, indices_or_sections=2, axis=-1)

    # 计算DBP分支的损失（例如，使用SNR损失）
    snr_loss = si_snr(jnp.abs(output_dbp.squeeze()), jnp.abs(aligned_x))

    # 计算神经网络分支的损失（例如，残差的均方误差）
    residual = output_dbp.squeeze()  # 假设残差为一维
    residual_loss = si_snr(jnp.abs(residual - (aligned_x - output_dbp.squeeze())), jnp.abs(aligned_x))
    # 总损失为两个损失的加权和
    alpha = 1.0  # 根据需要调整权重
    beta = 1.0
    total_loss = alpha * snr_loss + beta * residual_loss

    return total_loss, updated_state    

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


# def test(model: Model,
#          params: Dict,
#          data: gdat.Input,
#          eval_range: tuple=(300000, -20000),
#          metric_fn=comm.qamqot):
#     ''' testing, a simple forward pass

#         Args:
#             model: Model namedtuple return by `model_init`
#         data: dataset
#         eval_range: interval which QoT is evaluated in, assure proper eval of steady-state performance
#         metric_fn: matric function, comm.snrstat for global & local SNR performance, comm.qamqot for
#             BER, Q, SER and more metrics.

#         Returns:
#             evaluated matrics and equalized symbols
#     '''

#     state, aux, const, sparams = model.initvar[1:]
#     aux = core.dict_replace(aux, {'truth': data.x})
#     if params is None:
#       params = model.initvar[0]

#     z, _ = jit(model.module.apply,
#                backend='cpu')({
#                    'params': util.dict_merge(params, sparams),
#                    'aux_inputs': aux,
#                    'const': const,
#                    **state
#                }, core.Signal(data.y))
#     metric = metric_fn(z.val,
#                        data.x[z.t.start:z.t.stop],
#                        scale=np.sqrt(10),
#                        eval_range=eval_range)
#     return metric, z

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

def test(model: Model,
         params: Dict,
         data: gdat.Input,
         eval_range: tuple = (300000, -20000),
         metric_fn=comm.qamqot):
    ''' Testing function with ensemble learning '''

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

    # 拆分拼接的输出
    output_dbp, output_nn = jnp.split(z.val, indices_or_sections=2, axis=-1)

    # 对齐原始信号
    aligned_x = data.x[z.t_start:z.t_stop]

    # 确保输出形状一致
    output_dbp = output_dbp.squeeze()[:aligned_x.shape[0]]
    output_nn = output_nn.squeeze()[:aligned_x.shape[0]]

    # 定义权重，可以根据模型性能调整
    weight_dbp = 0.6
    weight_nn = 0.4

    # 加权平均融合
    output = weight_dbp * output_dbp + weight_nn * output_nn

    # 计算指标
    metric = metric_fn(output,
                       aligned_x,
                       scale=np.sqrt(10),
                       eval_range=eval_range)
    return metric, z

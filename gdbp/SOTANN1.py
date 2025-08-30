# =========================================
# QAL：商空间相位对齐 + 停止反传 + 取消 teacher-forcing
# （保持原网络结构与命名，可复用旧 checkpoint）
# =========================================
from typing import Any, Union, Optional, Tuple
from collections import namedtuple
from functools import partial
from tqdm.auto import tqdm

import numpy as np
import jax
from jax import numpy as jnp, random, jit, value_and_grad, lax
import jax.nn as jnn
import flax
from flax import linen as nn

from commplax import util, comm, cxopt, op, optim
from commplax.module import core, layer

# ---------------------------
# 类型别名 & Model 容器
# ---------------------------
Array = Any
Dict  = Union[dict, flax.core.FrozenDict]
Model = namedtuple('Model', 'module initvar overlaps name')

# ---------------------------
# 与你原来一致的模块构建（保持命名）
# ---------------------------
def make_base_module(steps: int = 3,
                     dtaps: int = 261,
                     ntaps: int = 41,
                     rtaps: int = 61,
                     init_fn: tuple = (core.delta, core.gauss),
                     w0=0.,
                     mode: str = 'train'):
    """
    不改结构：含两个分支（FDBP + FDBP1），FOEAf1/FOEAf 与 RConv1/RConv 命名保持不变。
    """
    _assert_taps(dtaps, ntaps, rtaps)
    d_init, n_init = init_fn

    if mode == 'train':
        mimo_train = True
    elif mode == 'test':
        mimo_train = cxopt.piecewise_constant([200000], [True, False])
    else:
        raise ValueError('invalid mode %s' % mode)

    # 串联 FDBP 分支
    fdbp_series = layer.Serial(
        layer.FDBP(steps=steps, dtaps=dtaps, ntaps=ntaps,
                   d_init=d_init, n_init=n_init, name='fdbp1'),
        layer.BatchPowerNorm(mode=mode),
        layer.MIMOFOEAf(name='FOEAf1',
                        w0=w0, train=mimo_train,
                        preslicer=core.conv1d_slicer(rtaps),
                        foekwargs={}),
        layer.vmap(layer.Conv1d)(name='RConv1', taps=rtaps),
        layer.MIMOAF(train=mimo_train),
        name='fdbp_series'
    )

    # 原有串行分支（保留 FDBP1）
    serial_branch = layer.Serial(
        layer.FDBP1(steps=steps, dtaps=dtaps, ntaps=ntaps,
                    d_init=d_init, n_init=n_init),
        layer.BatchPowerNorm(mode=mode),
        layer.MIMOFOEAf(name='FOEAf',
                        w0=w0, train=mimo_train,
                        preslicer=core.conv1d_slicer(rtaps),
                        foekwargs={}),
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
    """保持你原来的奇数 tap 约束"""
    assert dtaps % sps, f'dtaps must be odd number, got {dtaps} instead'
    assert ntaps % sps, f'ntaps must be odd number, got {ntaps} instead'
    assert rtaps % sps, f'rtaps must be odd number, got {rtaps} instead'

# ---------------------------
# FDBP 初始化（与你原实现一致）
# ---------------------------
def fdbp_init(a: dict,
              xi: float = 1.1,
              steps: Optional[int] = None):
    """
    返回 (d_init, n_init)
    """
    def d_init(key, shape, dtype=jnp.complex64):
        dtaps = shape[0]
        d0, _ = comm.dbp_params(
            a['samplerate'],
            a['distance'] / a['spans'],
            a['spans'],
            dtaps,
            a['lpdbm'] - 3,  # 输入功率已在 dataloader 归一为 2
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

# ---------------------------
# 模型初始化（与你原实现兼容）
# ---------------------------
def model_init(data,
               base_conf: dict,
               sparams_flatkeys: list,
               n_symbols: int = 4000,
               sps : int = 2,
               name='Model'):
    """
    不改接口；保持返回 (Model.module, initvar, overlaps, name)
    """
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

# ---------------------------
# 任务一致：Bit-BCE & 常量
# ---------------------------
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

def _bit_bce_loss_16qam(pred_sym: Array, true_sym: Array) -> Array:
    logits = -jnp.square(jnp.abs(pred_sym[..., None] - CONST_16QAM))     # [N,16]
    logp   = logits - jnn.logsumexp(logits, axis=-1, keepdims=True)
    probs  = jnp.exp(logp)                                               # [N,16]
    p1     = (probs @ BIT_MAP)                                           # [N,4]
    p0     = 1.0 - p1
    idx    = jnp.argmin(jnp.square(jnp.abs(true_sym[..., None] - CONST_16QAM)), axis=-1)
    bits   = BIT_MAP[idx]                                                # [N,4]
    bce    = -(bits*jnp.log(p1+1e-12) + (1.-bits)*jnp.log(p0+1e-12))
    return (bce * BIT_WEIGHTS).mean()

# ---------------------------
# 核心：QAL 损失（仅移除相位 + stop-grad；不对齐幅度）
# ---------------------------
def loss_fn(module: layer.Layer,
            params: Dict,
            state : Dict,
            y     : Array,
            x     : Array,
            aux   : Dict,
            const : Dict,
            sparams: Dict,
            β_ce : float = 0.5,
            λ_kl : float = 1e-4):
    """
    QAL：把输出与真值在 U(1) 上对齐（只去全局相位），
    对齐相位的因子不反传；幅度不对齐，交给 BN/MIMOAF。
    在去相位后的 y_al 上算 EVM、SI-SNR、Bit-BCE；
    KL 仍作用在未对齐 yhat 上（能量正则）。
    """
    params_net = util.dict_merge(params, sparams)
    z_out, state_new = module.apply(
        {'params': params_net, 'aux_inputs': aux, 'const': const, **state},
        core.Signal(y)
    )

    x_ref = x[z_out.t.start:z_out.t.stop]    # 参考符号
    yhat  = z_out.val                        # 预测复符号

    # —— 相位闭式对齐 + 停止反传 ——（不做幅度 alpha）
    zc   = jnp.vdot(yhat.reshape(-1), x_ref.reshape(-1))  # <ŷ, x>
    p    = zc / (jnp.abs(zc) + 1e-8)                      # e^{jφ*}
    psg  = lax.stop_gradient(jnp.conj(p))                 # ★ 不反传 φ*
    y_al = yhat * psg                                     # 去相位输出

    # —— 任务一致计分（在 y_al 上）——
    # 1) EVM（Q² 代理）
    evm = (jnp.mean(jnp.abs(y_al - x_ref)**2) /
           (jnp.mean(jnp.abs(x_ref)**2) + 1e-8))
    # 2) SI-SNR（复数版）
    t = x_ref.reshape(-1); e = y_al.reshape(-1)
    a = jnp.vdot(t, e) / (jnp.vdot(t, t) + 1e-8)
    s = a * t; err = e - s
    snr = -10.0 * jnp.log10(
        (jnp.real(jnp.vdot(s, s)) + 1e-8) /
        (jnp.real(jnp.vdot(err, err)) + 1e-8)
    )
    # 3) Bit-BCE
    bit_bce = _bit_bce_loss_16qam(y_al, x_ref)

    # 4) 信息瓶颈 KL（在未对齐 yhat）
    kl_ib = 0.5 * jnp.mean(jnp.square(jnp.abs(yhat)))

    loss_main  = snr + 0.1 * evm + β_ce * bit_bce
    total_loss = loss_main + λ_kl * kl_ib
    return total_loss, state_new

# ---------------------------
# 单步更新
# ---------------------------
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
        loss_fn, argnums=1, has_aux=True)(module, params, module_state, y, x,
                                          aux, const, sparams)
    opt_state = opt.update_fn(i, grads, opt_state)
    return loss, opt_state, module_state

# ---------------------------
# 批生成（与你原实现一致）
# ---------------------------
def get_train_batch(ds,
                    batchsize: int,
                    overlaps: int,
                    sps: int = 2):
    flen  = batchsize + overlaps
    fstep = batchsize
    ds_y  = op.frame_gen(ds.y, flen * sps, fstep * sps)
    ds_x  = op.frame_gen(ds.x, flen, fstep)
    n_batches = op.frame_shape(ds.x.shape, flen, fstep)[0]
    return n_batches, zip(ds_y, ds_x)

# ---------------------------
# 训练：取消把 truth 喂给 FOE/MIMOAF（避免“吃梯度”）
# ---------------------------
def train(model: Model,
          data,
          batch_size: int = 500,
          n_iter = None,
          opt: optim.Optimizer = optim.adam(
              optim.piecewise_constant([500, 1000], [1e-4, 1e-5, 1e-6])
          )):
    params, module_state, aux, const, sparams = model.initvar
    opt_state = opt.init_fn(params)

    n_batch, batch_gen = get_train_batch(data, batch_size, model.overlaps)
    n_iter = n_batch if n_iter is None else min(n_iter, n_batch)

    for i, (y, x) in tqdm(enumerate(batch_gen),
                          total=n_iter, desc='training', leave=False):
        if i >= n_iter:
            break
        # ★ 不要 teacher-forcing（不要把 truth 喂给 aux）
        # aux = core.dict_replace(aux, {'truth': x})
        loss, opt_state, module_state = update_step(
            model.module, opt, i, opt_state, module_state, y, x, aux, const, sparams
        )
        yield loss, opt.params_fn(opt_state), module_state

# ---------------------------
# 测试：同样不喂 truth（与训练一致）
# ---------------------------
def test(model: Model,
         params: Dict,
         data,
         eval_range: Tuple[int, int]=(300000, -20000),
         metric_fn=comm.qamqot):
    state, aux, const, sparams = model.initvar[1:]
    # ★ 测试同样不喂 truth
    # aux = core.dict_replace(aux, {'truth': data.x})
    if params is None:
        params = model.initvar[0]

    z, _ = jit(model.module.apply, backend='cpu')({
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

# ---------------------------
# （可选）推平工具：与原实验兼容
# ---------------------------
def equalize_dataset(model_te, params, state_bundle, data):
    module_state, aux, const, sparams = state_bundle
    z,_ = jax.jit(model_te.module.apply, backend='cpu')(
        {'params': util.dict_merge(params, sparams),
         'aux_inputs': aux, 'const': const, **module_state},
        core.Signal(data.y))

    start, stop = z.t.start, z.t.stop
    z_eq  = np.asarray(z.val[:,0])            # equalized
    s_ref = np.asarray(data.x)[start:stop,0]  # 保持原尺度
    return z_eq, s_ref

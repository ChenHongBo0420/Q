# =========================================
# SOTANN1 with Gradient Routing on Quotient Space
# 保留原有结构/命名；只新增“梯度路由对齐”与形状护栏
# =========================================
from typing import Any, Optional, Union, Tuple
from collections import namedtuple
from functools import partial

import numpy as np
import jax
from jax import numpy as jnp, random, jit, value_and_grad, lax
import flax
from flax import linen as nn
from flax.core import freeze

from commplax import util, comm, cxopt, op, optim
from commplax.module import core, layer

# -----------------------------------------------------------------------------
# 基本别名 & Model 容器
# -----------------------------------------------------------------------------
Array = Any
Dict  = Union[dict, flax.core.FrozenDict]
Model = namedtuple('Model', 'module initvar overlaps name')

# -----------------------------------------------------------------------------
# 16QAM 常量 & Bit-BCE
# -----------------------------------------------------------------------------
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
    logp   = logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True)   # [N,16]
    probs  = jnp.exp(logp)
    p1     = (probs @ BIT_MAP)                                           # [N,4]
    p0     = 1.0 - p1
    idx    = jnp.argmin(jnp.square(jnp.abs(true_sym[..., None] - CONST_16QAM)), axis=-1)
    bits   = BIT_MAP[idx]                                                # [N,4]
    bce    = -(bits*jnp.log(p1+1e-12) + (1.-bits)*jnp.log(p0+1e-12))
    return (bce * BIT_WEIGHTS).mean()

# -----------------------------------------------------------------------------
# [ADDED] 形状护栏：统一时间长度与通道维（避免 vdot 维度不齐）
# -----------------------------------------------------------------------------
def _match_y_x(yhat: jnp.ndarray, x_ref: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    统一 yhat 与 x_ref 的时间长度与通道维：
      - 截齐时间长度 L = min(Ty, Tx)
      - 若 x_ref 无通道维或为 1 通道，广播到 yhat 的通道数
      - 返回形状完全一致的 (yv, xv)
    """
    # squeeze 单通道尾维
    def _squeeze_last1(a):
        return a[..., 0] if (a.ndim >= 2 and a.shape[-1] == 1) else a

    yv = _squeeze_last1(yhat)
    xv = _squeeze_last1(x_ref)

    # 时间维对齐
    Ty = yv.shape[0]
    Tx = xv.shape[0]
    L  = Ty if Tx is None else min(Ty, Tx)
    yv = yv[:L]
    xv = xv[:L]

    # 通道维对齐
    if yv.ndim == 1 and xv.ndim == 1:
        # 都是 [L]，直接返回
        return yv, xv
    if yv.ndim == 2 and xv.ndim == 1:
        # y=[L,C], x=[L] → 把 x 广播到 C 通道
        C = yv.shape[-1]
        xv = jnp.repeat(xv[:, None], C, axis=1)
        return yv, xv
    if yv.ndim == 2 and xv.ndim == 2:
        Cy, Cx = yv.shape[-1], xv.shape[-1]
        if Cx == Cy:
            return yv, xv
        if Cx == 1:
            xv = jnp.repeat(xv, Cy, axis=1)
            return yv, xv
        # 其它不匹配场景：退回只取前 min(Cy,Cx)
        C = min(Cy, Cx)
        return yv[:, :C], xv[:, :C]
    if yv.ndim == 1 and xv.ndim == 2:
        # 罕见：把 y 扩到多通道（复制）
        C = xv.shape[-1]
        yv = jnp.repeat(yv[:, None], C, axis=1)
        return yv, xv

    return yv, xv

# -----------------------------------------------------------------------------
# [ADDED] 梯度路由：自定义 VJP 的相位对齐（前向去相位；反向投影掉相位/增益梯度）
# -----------------------------------------------------------------------------
@jax.custom_vjp
def align_phase_route(yhat: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """
    前向：仅去全局相位，y_al = yhat * conj(p),  p = <yhat,x>/|<yhat,x>|
    反向：把梯度映回 yhat 坐标后，剔除落在 span{yhat, i*yhat} 的分量
          （只保留“物理水平”梯度，屏蔽增益/相位的规范分量）。
    """
    eps = 1e-8
    zc = jnp.vdot(yhat.reshape(-1), x.reshape(-1))
    p  = zc / (jnp.abs(zc) + eps)
    return yhat * jnp.conj(p)

def _align_phase_fwd(yhat: jnp.ndarray, x: jnp.ndarray):
    eps = 1e-8
    zc = jnp.vdot(yhat.reshape(-1), x.reshape(-1))
    p  = zc / (jnp.abs(zc) + eps)
    y_al = yhat * jnp.conj(p)
    # 关键：把 x 一并放进残差，供反向阶段 zeros_like(x) 使用
    return y_al, (yhat, x, p)

def _align_phase_bwd(res, g_al):
    yhat, x, p = res
    eps = 1e-8

    # 把梯度从 y_al 坐标映回 yhat：y_al = conj(p)*yhat  ⇒  dyhat = p * dy_al
    g_back = p * g_al

    # 规范基 span{yhat, i*yhat} 上的分量去掉，只回传水平分量
    yh = yhat.reshape(-1)
    g  = g_back.reshape(-1)
    n2 = jnp.real(jnp.vdot(yh, yh)) + eps
    u1 = yh / jnp.sqrt(n2)      # 幅度方向
    u2 = 1j * u1                # 相位方向

    c1 = jnp.vdot(u1, g)
    c2 = jnp.vdot(u2, g)
    g_vert  = u1 * c1 + u2 * c2
    g_horiz = g - g_vert

    g_yhat = g_horiz.reshape(yhat.shape)
    g_x    = jnp.zeros_like(x)  # 不把梯度推回真值分支
    return (g_yhat, g_x)

align_phase_route.defvjp(_align_phase_fwd, _align_phase_bwd)

# -----------------------------------------------------------------------------
# 基础模块（保持原结构与命名）
# -----------------------------------------------------------------------------
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
        raise ValueError(f'invalid mode {mode}')

    fdbp_series = layer.Serial(
        layer.FDBP(steps=steps, dtaps=dtaps, ntaps=ntaps,
                   d_init=d_init, n_init=n_init, name='fdbp1'),
        layer.BatchPowerNorm(mode=mode),
        layer.MIMOFOEAf(name='FOEAf1', w0=w0, train=mimo_train,
                        preslicer=core.conv1d_slicer(rtaps), foekwargs={}),
        layer.vmap(layer.Conv1d)(name='RConv1', taps=rtaps),
        layer.MIMOAF(train=mimo_train),
        name='fdbp_series'
    )

    serial_branch = layer.Serial(
        layer.FDBP1(steps=steps, dtaps=dtaps, ntaps=ntaps,
                    d_init=d_init, n_init=n_init),
        layer.BatchPowerNorm(mode=mode),
        layer.MIMOFOEAf(name='FOEAf', w0=w0, train=mimo_train,
                        preslicer=core.conv1d_slicer(rtaps), foekwargs={}),
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
    assert dtaps % sps, f'dtaps must be odd number, got {dtaps} instead'
    assert ntaps % sps, f'ntaps must be odd number, got {ntaps} instead'
    assert rtaps % sps, f'rtaps must be odd number, got {rtaps} instead'

# -----------------------------------------------------------------------------
# FDBP 初始化（保持原逻辑）
# -----------------------------------------------------------------------------
def fdbp_init(a: dict, xi: float = 1.1, steps: Optional[int] = None):
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

# -----------------------------------------------------------------------------
# 模型初始化（增强鲁棒：af_state/aux/const 缺失时兜底）
# -----------------------------------------------------------------------------
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

    # 兼容无 af_state/aux_inputs/const 的版本
    af_state   = v0.get('af_state',   freeze({}))
    aux_inputs = v0.get('aux_inputs', freeze({}))
    const      = v0.get('const',      freeze({}))
    state = {'af_state': af_state}

    return Model(mod, (params, state, aux_inputs, const, sparams), ol, name)

# -----------------------------------------------------------------------------
# 一些度量（与原逻辑一致/相近）
# -----------------------------------------------------------------------------
def si_snr_complex(tx: jnp.ndarray, rx: jnp.ndarray, eps=1e-8) -> jnp.ndarray:
    t = tx.reshape(-1); e = rx.reshape(-1)
    a = jnp.vdot(t, e) / (jnp.vdot(t, t) + eps)
    s = a * t
    err = e - s
    return -10.0 * jnp.log10(
        (jnp.real(jnp.vdot(s, s)) + eps) / (jnp.real(jnp.vdot(err, err)) + eps)
    )

def evm_norm(tx: jnp.ndarray, rx: jnp.ndarray, eps=1e-8) -> jnp.ndarray:
    num = jnp.mean(jnp.abs(rx - tx)**2)
    den = jnp.mean(jnp.abs(tx)**2) + eps
    return num / den

# -----------------------------------------------------------------------------
# 损失函数：用“梯度路由相位对齐”替代原相位对齐；其余不动
# -----------------------------------------------------------------------------
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

    params_net = util.dict_merge(params, sparams)

    # 前向
    z_out, state_new = module.apply(
        {'params': params_net, 'aux_inputs': aux, 'const': const, **state},
        core.Signal(y))

    # 对齐 x 与 y 的形状（时间 & 通道）
    yhat  = z_out.val
    x_ref = x[z_out.t.start:z_out.t.stop]
    yv, xv = _match_y_x(yhat, x_ref)

    # [ADDED] 商空间相位对齐 + 梯度路由（只回传水平梯度）
    y_al = align_phase_route(yv, xv)

    # 任务损失（在对齐后计算）
    evm    = evm_norm(xv, y_al)
    snr    = si_snr_complex(xv, y_al)
    bit_ce = _bit_bce_loss_16qam(y_al, xv)

    # 信息瓶颈（未对齐输出，限制能量）
    kl_ib  = 0.5 * jnp.mean(jnp.square(jnp.abs(yv)))

    loss_main  = snr + 0.1 * evm + β_ce * bit_ce
    total_loss = loss_main + λ_kl * kl_ib
    return total_loss, state_new

# -----------------------------------------------------------------------------
# 单步更新（保持原接口；jit）
# -----------------------------------------------------------------------------
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
        loss_fn, argnums=1, has_aux=True)(
            module, params, module_state, y, x, aux, const, sparams)
    opt_state = opt.update_fn(i, grads, opt_state)
    return loss, opt_state, module_state

# -----------------------------------------------------------------------------
# 训练批生成（与原一致）
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# 训练（保留原“把 truth 喂给 aux”的写法，如需禁用可注释一行）
# -----------------------------------------------------------------------------
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

    for i, (y, x) in enumerate(batch_gen):
        if i >= n_iter:
            break
        # 如需避免 teacher-forcing，请注释下一行
        aux = core.dict_replace(aux, {'truth': x})

        loss, opt_state, module_state = update_step(
            model.module, opt, i, opt_state, module_state, y, x, aux, const, sparams
        )
        yield loss, opt.params_fn(opt_state), module_state

# -----------------------------------------------------------------------------
# 测试（保持原指标评估；对 y_eval 做闭式相位+增益对齐以评 QoT）
# -----------------------------------------------------------------------------
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

    yhat = z.val
    xref = data.x[z.t.start:z.t.stop]
    yv, xv = _match_y_x(yhat, xref)

    # 评测对齐（相位 + 闭式幅度），只用于 metric，不影响训练
    zc  = jnp.vdot(yv.reshape(-1), xv.reshape(-1))
    p   = zc / (jnp.abs(zc) + 1e-12)
    yph = yv * jnp.conj(p)
    num = jnp.real(jnp.vdot(yph.reshape(-1), xv.reshape(-1)))
    den = jnp.real(jnp.vdot(yph.reshape(-1), yph.reshape(-1))) + 1e-12
    y_eval = (num/den) * yph

    metric = metric_fn(y_eval, xv, scale=np.sqrt(10), eval_range=eval_range)
    return metric, z

# -----------------------------------------------------------------------------
# 等化器输出（与原一致）
# -----------------------------------------------------------------------------
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

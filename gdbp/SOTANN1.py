# =========================================
# SOTANN1 with Quotient-Space + Gradient Routing
# 保留原网络结构与命名；仅新增：
#  - 商空间相位对齐（custom_vjp，反向去掉相位/增益分量）
#  - 梯度路由（按模块缩放，JAX-safe，无 Python if）
#  - 全局梯度范数裁剪
#  - 形状护栏（对齐 y/x 的时间与通道）
# =========================================
from typing import Any, Optional, Union, Tuple, List
from collections import namedtuple
from functools import partial

import numpy as np
import jax
from jax import numpy as jnp, random, jit, value_and_grad, lax
import jax.nn as jnn
import flax
from flax import linen as nn
from flax.core import freeze
from flax.traverse_util import flatten_dict, unflatten_dict

from commplax import util, comm, cxopt, op, optim
from commplax.module import core, layer

# -----------------------------------------------------------------------------
# 别名 & Model 容器
# -----------------------------------------------------------------------------
Array = Any
Dict  = Union[dict, flax.core.FrozenDict]
Model = namedtuple('Model', 'module initvar overlaps name')

# -----------------------------------------------------------------------------
# 常量
# -----------------------------------------------------------------------------
EPS = 1e-8
CLIP_NORM = 1.0          # 全局梯度裁剪上限
TF_ENABLE = True         # 是否将 truth 喂给 aux（teacher-forcing）；如需一致训练/测试，可设 False

# -----------------------------------------------------------------------------
# 16QAM & Bit-BCE
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
    """Bit-BCE（加权）。pred_sym/true_sym: [T] 或 [T,C] 复数。"""
    logits = -jnp.square(jnp.abs(pred_sym[..., None] - CONST_16QAM))   # [...,16]
    logp   = logits - jnn.logsumexp(logits, axis=-1, keepdims=True)
    probs  = jnp.exp(logp)
    p1     = (probs @ BIT_MAP)                                         # [...,4]
    p0     = 1.0 - p1

    idx  = jnp.argmin(jnp.square(jnp.abs(true_sym[..., None] - CONST_16QAM)), axis=-1)
    bits = BIT_MAP[idx]                                                # [...,4]

    bce  = -(bits * jnp.log(p1 + 1e-12) + (1. - bits) * jnp.log(p0 + 1e-12))
    return (bce * BIT_WEIGHTS).mean()

# -----------------------------------------------------------------------------
# 形状护栏：对齐 yhat 与 x_ref 的时间长度与通道维
# -----------------------------------------------------------------------------
def _match_y_x(yhat: jnp.ndarray, x_ref: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    - 截齐时间长度 L = min(Ty, Tx)
    - 若 x_ref 是单通道而 yhat 多通道，则广播 x_ref
    - 返回形状一致的 (yv, xv)；均为 [L] 或 [L,C]
    """
    # 压掉尾部 size=1 的通道维
    def _squeeze_last1(a):
        return a[..., 0] if (a.ndim >= 2 and a.shape[-1] == 1) else a

    yv = _squeeze_last1(yhat)
    xv = _squeeze_last1(x_ref)

    Ty = yv.shape[0]
    Tx = xv.shape[0]
    L  = Ty if Tx is None else min(Ty, Tx)
    yv = yv[:L]
    xv = xv[:L]

    # 通道维处理
    if yv.ndim == 1 and xv.ndim == 1:
        return yv, xv
    if yv.ndim == 2 and xv.ndim == 1:
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
        C = min(Cy, Cx)
        return yv[:, :C], xv[:, :C]
    if yv.ndim == 1 and xv.ndim == 2:
        C = xv.shape[-1]
        yv = jnp.repeat(yv[:, None], C, axis=1)
        return yv, xv
    return yv, xv

# -----------------------------------------------------------------------------
# 商空间相位对齐（自定义 VJP）：前向去相位，反向剔除增益/相位的规范梯度
# -----------------------------------------------------------------------------
@jax.custom_vjp
def align_phase_route(yhat: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """
    前向：y_al = yhat * conj(p)，p = <yhat,x>/|<yhat,x>|
    反向：只回传“水平”梯度（去掉沿 span{yhat, i*yhat} 的分量）
    """
    zc = jnp.vdot(yhat.reshape(-1), x.reshape(-1))
    p  = zc / (jnp.abs(zc) + EPS)
    return yhat * jnp.conj(p)

def _align_phase_fwd(yhat: jnp.ndarray, x: jnp.ndarray):
    zc = jnp.vdot(yhat.reshape(-1), x.reshape(-1))
    p  = zc / (jnp.abs(zc) + EPS)
    y_al = yhat * jnp.conj(p)
    return y_al, (yhat, x, p)

def _align_phase_bwd(res, g_al):
    yhat, x, p = res
    # 从 y_al 坐标映回 yhat： y_al = conj(p) * yhat  ⇒ dyhat = p * dy_al
    g_back = p * g_al

    yh = yhat.reshape(-1)
    g  = g_back.reshape(-1)

    n2 = jnp.real(jnp.vdot(yh, yh)) + EPS
    u1 = yh / jnp.sqrt(n2)      # 幅度方向
    u2 = 1j * u1                # 相位方向

    c1 = jnp.vdot(u1, g)
    c2 = jnp.vdot(u2, g)
    g_vert  = u1 * c1 + u2 * c2             # 规范（增益/相位）梯度
    g_horiz = g - g_vert                    # 水平（物理）梯度

    g_yhat = g_horiz.reshape(yhat.shape)
    g_x    = jnp.zeros_like(x)              # 不把梯度推回真值分支
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
            a['samplerate'], a['distance'] / a['spans'], a['spans'],
            dtaps, a['lpdbm'] - 3, virtual_spans=steps)
        return d0[0, :, 0]

    def n_init(key, shape, dtype=jnp.float32):
        dtaps = shape[0]
        _, n0 = comm.dbp_params(
            a['samplerate'], a['distance'] / a['spans'], a['spans'],
            dtaps, a['lpdbm'] - 3, virtual_spans=steps)
        return xi * n0[0, 0, 0] * core.gauss(key, shape, dtype)

    return d_init, n_init

# -----------------------------------------------------------------------------
# 模型初始化（鲁棒获取集合：af_state/aux_inputs/const 可能缺失）
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

    af_state   = v0.get('af_state',   freeze({}))
    aux_inputs = v0.get('aux_inputs', freeze({}))
    const      = v0.get('const',      freeze({}))
    state = {'af_state': af_state}

    return Model(mod, (params, state, aux_inputs, const, sparams), ol, name)

# -----------------------------------------------------------------------------
# 指标
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
# 梯度裁剪
# -----------------------------------------------------------------------------
def _global_l2(tree):
    sqsum = 0.0
    for x in jax.tree_util.tree_leaves(tree):
        sqsum = sqsum + jnp.sum(jnp.square(jnp.asarray(x)))
    return jnp.sqrt(sqsum)

def _clip_by_global_norm(grads, clip_norm: float):
    gnorm = _global_l2(grads) + 1e-12
    scale = jnp.minimum(1.0, clip_norm / gnorm)
    return jax.tree_map(lambda g: g * scale, grads)

# -----------------------------------------------------------------------------
# 梯度路由（JAX-safe，无 Python 分支）
# -----------------------------------------------------------------------------
def _rule_factor(base_factor, i):
    """warmup：前 300 步→1.0；之后→base_factor。返回 0D DeviceArray。"""
    i = jnp.asarray(i)
    warm = jnp.where(i <= 300, 0.0, 1.0)
    return 1.0 + (base_factor - 1.0) * warm

def _rules_with_i(i) -> List[Tuple[Tuple[str, ...], jnp.ndarray]]:
    """
    返回 [(patterns, factor_device_array), ...]
    注意：patterns 需与实际 param 名匹配；若不匹配，等同于 1.0（不生效）。
    """
    return [
        # 放大非线性/FDBP 等“相位敏感块”的梯度（等变/物理层）
        (("fdbp", "FDBP", "fdbp1", "/n", "nonlinear", "n_filter"), _rule_factor(1.35, i)),
        # 轻抑制 MIMOAF/RConv 等“相位不变块”的梯度（不变/判决层）
        (("RConv", "RConv1", "MIMOAF"),                            _rule_factor(0.80, i)),
        # FOE/FOEAf 维持 1
        (("FOEAf", "FOEAf1"),                                      jnp.array(1.0)),
    ]

def _scale_grads_by_rules(grads: Dict, rules) -> Dict:
    flat = flatten_dict(grads, keep_empty_nodes=True)
    new_flat = {}
    for k, v in flat.items():
        name = "/".join(k)
        fac = 1.0
        for pats, f in rules:
            if any(p in name for p in pats):
                fac = fac * f  # f 是 0D DeviceArray，jit 安全
        new_flat[k] = v * fac
    return unflatten_dict(new_flat)

# -----------------------------------------------------------------------------
# 损失：商空间相位对齐（反向投影）＋ 分布一致性（Bit-BCE/EVM/SI-SNR）＋ IB-KL
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
    yhat  = z_out.val
    x_ref = x[z_out.t.start:z_out.t.stop]
    yv, xv = _match_y_x(yhat, x_ref)

    # 商空间相位对齐（反向仅回传水平梯度）
    y_al = align_phase_route(yv, xv)

    # 任务损失
    evm    = evm_norm(xv, y_al)
    snr    = si_snr_complex(xv, y_al)
    bit_ce = _bit_bce_loss_16qam(y_al, xv)

    # 信息瓶颈（未对齐输出，限制绝对能量）
    kl_ib  = 0.5 * jnp.mean(jnp.square(jnp.abs(yv)))

    loss_main  = snr + 0.1 * evm + β_ce * bit_ce
    total_loss = loss_main + λ_kl * kl_ib
    return total_loss, state_new

# -----------------------------------------------------------------------------
# 单步更新（jit）
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

    # 梯度路由（JAX-safe 系数）
    rules = _rules_with_i(i)
    grads = _scale_grads_by_rules(grads, rules)

    # 全局裁剪
    grads = _clip_by_global_norm(grads, CLIP_NORM)

    opt_state = opt.update_fn(i, grads, opt_state)
    return loss, opt_state, module_state

# -----------------------------------------------------------------------------
# 批生成（与原来一致）
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
# 训练（默认保留 TF；若要关闭，把 TF_ENABLE 设为 False）
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
        if TF_ENABLE:
            aux = core.dict_replace(aux, {'truth': x})

        loss, opt_state, module_state = update_step(
            model.module, opt, i, opt_state, module_state, y, x, aux, const, sparams
        )
        yield loss, opt.params_fn(opt_state), module_state

# -----------------------------------------------------------------------------
# 测试（评测时闭式相位+幅度对齐，仅用于 metric）
# -----------------------------------------------------------------------------
def test(model: Model,
         params: Dict,
         data,
         eval_range: tuple=(300000, -20000),
         metric_fn=comm.qamqot):
    state, aux, const, sparams = model.initvar[1:]
    if TF_ENABLE:
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

    # 评测对齐（相位 + 闭式幅度）
    zc  = jnp.vdot(yv.reshape(-1), xv.reshape(-1))
    p   = zc / (jnp.abs(zc) + 1e-12)
    yph = yv * jnp.conj(p)
    num = jnp.real(jnp.vdot(yph.reshape(-1), xv.reshape(-1)))
    den = jnp.real(jnp.vdot(yph.reshape(-1), yph.reshape(-1))) + 1e-12
    y_eval = (num/den) * yph

    metric = metric_fn(y_eval, xv, scale=np.sqrt(10), eval_range=eval_range)
    return metric, z

# -----------------------------------------------------------------------------
# 等化输出（保持原样）
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

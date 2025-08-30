from typing import Any, Union
from jax import numpy as jnp, jit, value_and_grad, lax
import jax.nn as jnn
import flax
from commplax import util, comm, cxopt, op, optim
from commplax.module import core, layer
from tqdm.auto import tqdm

Array = Any
Dict  = Union[dict, flax.core.FrozenDict]

# ---------- 常量 & Bit-BCE（保持你之前的 16QAM 设定） ----------
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

# ---------- 核心：商空间相位对齐（仅移除全局相位）+ stop-grad ----------
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
    QAL：仅移除全局相位（U(1) 商），并对相位解 stop-gradient；
         在去相位后的输出 y_al 上计算 EVM / SI-SNR / Bit-BCE；
         KL 仍作用在未对齐输出 yhat（能量正则）。
    """
    params_net = util.dict_merge(params, sparams)
    z_out, state_new = module.apply(
        {'params': params_net, 'aux_inputs': aux, 'const': const, **state},
        core.Signal(y)
    )

    x_ref = x[z_out.t.start:z_out.t.stop]    # 参考符号
    yhat  = z_out.val                        # 预测复符号

    # —— 相位闭式对齐 + 停止反传（不做幅度 alpha，对幅度留给 BN/MIMOAF）——
    zc   = jnp.vdot(yhat.reshape(-1), x_ref.reshape(-1))   # <ŷ, x>
    p    = zc / (jnp.abs(zc) + 1e-8)                       # e^{jφ*}
    psg  = lax.stop_gradient(jnp.conj(p))                  # ★ 不反传 φ*
    y_al = yhat * psg                                      # 去相位后的输出

    # —— 任务一致计分（在 y_al 上）——
    # (1) EVM（Q² 代理）
    evm = (jnp.mean(jnp.abs(y_al - x_ref)**2) /
           (jnp.mean(jnp.abs(x_ref)**2) + 1e-8))

    # (2) SI-SNR（复数版）
    t = x_ref.reshape(-1); e = y_al.reshape(-1)
    a = jnp.vdot(t, e) / (jnp.vdot(t, t) + 1e-8)
    s = a * t; err = e - s
    snr = -10.0 * jnp.log10(
        (jnp.real(jnp.vdot(s, s)) + 1e-8) /
        (jnp.real(jnp.vdot(err, err)) + 1e-8)
    )

    # (3) Bit-BCE（对齐后星座）
    bit_bce = _bit_bce_loss_16qam(y_al, x_ref)

    # (4) 信息瓶颈 KL（作用在未对齐 yhat）
    kl_ib = 0.5 * jnp.mean(jnp.square(jnp.abs(yhat)))

    loss_main  = snr + 0.1 * evm + β_ce * bit_bce
    total_loss = loss_main + λ_kl * kl_ib
    return total_loss, state_new

# ---------- 单步更新：无需改动逻辑，仅接上新的 loss_fn ----------
@jit(static_argnums=(0, 1), backend='cpu')
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

# ---------- 训练/测试：取消把 truth 喂给 FOE/MIMOAF（避免“吃梯度式对齐”） ----------
def train(model,
          data,
          batch_size: int = 500,
          n_iter = None,
          opt: optim.Optimizer = optim.adam(optim.piecewise_constant([500, 1000], [1e-4, 1e-5, 1e-6]))):
    params, module_state, aux, const, sparams = model.initvar
    opt_state = opt.init_fn(params)

    # 生成批（沿用你原始实现）
    flen = batch_size + model.overlaps
    fstep = batch_size
    ds_y = op.frame_gen(data.y, flen * 2, fstep * 2)  # sps=2
    ds_x = op.frame_gen(data.x, flen, fstep)
    n_batch = op.frame_shape(data.x.shape, flen, fstep)[0]
    n_iter = n_batch if n_iter is None else min(n_iter, n_batch)

    for i, (y, x) in tqdm(enumerate(zip(ds_y, ds_x)), total=n_iter, desc='training', leave=False):
        if i >= n_iter: break
        # ★ 不要 teacher-forcing（不要把 truth 喂给 aux）
        # aux = core.dict_replace(aux, {'truth': x})
        loss, opt_state, module_state = update_step(model.module, opt, i, opt_state,
                                                   module_state, y, x, aux,
                                                   const, sparams)
        yield loss, opt.params_fn(opt_state), module_state

def test(model,
         params,
         data,
         eval_range: tuple=(300000, -20000),
         metric_fn=comm.qamqot):
    state, aux, const, sparams = model.initvar[1:]
    # ★ 测试同样不喂 truth，保持与训练一致
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
                       scale=jnp.sqrt(10.0),
                       eval_range=eval_range)
    return metric, z

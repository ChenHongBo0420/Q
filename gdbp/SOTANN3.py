# Copyright 2021 The Commplax Authors.
# Modified: Split-loss + Gradient Routing for invariant (back-end) vs equivariant (front-end)
# Drop-in replacement: same public API (train/test/equalize_dataset); add test_ext for GMI/NGMI.

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

# -------------------- Types --------------------
Model = namedtuple('Model', 'module initvar overlaps name')
Array = Any
Dict  = Union[dict, flax.core.FrozenDict]
# ========= 1) 并行双分支的基础模块（A=FDBP 主；B=CDC-like 副），不在内部融合 =========
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

    # A：FDBP 主分支（带 N 滤波，等变/相位敏）
    fdbp_series = layer.Serial(
        layer.FDBP(steps=steps,
                   dtaps=dtaps,
                   ntaps=ntaps,
                   d_init=d_init,
                   n_init=n_init,
                   name='FDBP_A'),
        layer.BatchPowerNorm(mode=mode),
        layer.MIMOFOEAf(name='FOEAf_A',
                        w0=w0,
                        train=mimo_train,
                        preslicer=core.conv1d_slicer(rtaps),
                        foekwargs={}),
        layer.vmap(layer.Conv1d)(name='RConv_A', taps=rtaps),
        layer.MIMOAF(train=mimo_train),
        name='branch_A'
    )

    # B：CDC-like 副分支（N taps=1 近似无非线性，偏相位不变）
    serial_branch = layer.Serial(
        layer.FDBP1(steps=steps,
                    dtaps=dtaps,
                    ntaps=1,          # ★ 关键：CDC-like
                    d_init=d_init,
                    n_init=n_init,
                    name='FDBP_B_CDC'),
        layer.BatchPowerNorm(mode=mode),
        layer.MIMOFOEAf(name='FOEAf_B',
                        w0=w0,
                        train=mimo_train,
                        preslicer=core.conv1d_slicer(rtaps),
                        foekwargs={}),
        layer.vmap(layer.Conv1d)(name='RConv_B', taps=rtaps),
        layer.MIMOAF(train=mimo_train),
        name='branch_B'
    )

    # 不做内部 FanInMean，直接把两路输出丢出去
    base = layer.Serial(
        layer.FanOut(num=2),
        layer.Parallel(
            fdbp_series,
            serial_branch
        )
    )
    return base


# ========= 2) model_init：兼容 tuple 输出，取两路时间交集算 overlaps =========
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
    # z0 现在是 (sigA, sigB)
    if isinstance(z0, (tuple, list)):
        tA, tB = z0[0].t, z0[1].t
        t_start = max(tA.start, tB.start)
        t_stop  = min(tA.stop,  tB.stop)
        ol = t_start - t_stop
    else:
        ol = z0.t.start - z0.t.stop

    sparams, params = util.dict_split(v0['params'], sparams_flatkeys)
    state = v0['af_state']
    aux   = v0['aux_inputs']
    const = v0['const']
    return Model(mod, (params, state, aux, const, sparams), ol, name)


# ========= 3) 路由用的两个小辅助 =========
from jax import lax

def _align_phase_only(yhat, x_ref, tail=16384, eps=1e-8):
    """仅去全局相位（估计相位 stop-grad），保持幅度与时间结构。"""
    yv = yhat.reshape(-1); xv = x_ref.reshape(-1)
    W  = min(tail, yv.shape[0], xv.shape[0])
    ys = yv[-W:]; xs = xv[-W:]
    z  = jnp.vdot(ys, xs)
    p  = z / (jnp.abs(z) + eps)  # e^{jφ*}
    p  = lax.stop_gradient(p)    # ★ 相位估计不回传
    return yhat * jnp.conj(p)

def _crop_to_intersection(sigA, sigB, x_full):
    """对齐两支输出的时间交集，并裁剪 x 到同一窗口。"""
    tA, tB = sigA.t, sigB.t
    start = max(tA.start, tB.start)
    stop  = min(tA.stop,  tB.stop)

    yA = sigA.val[start - tA.start : sigA.val.shape[0] + stop - tA.stop]
    yB = sigB.val[start - tB.start : sigB.val.shape[0] + stop - tB.stop]
    xR = x_full[start : x_full.shape[0] + stop]
    tI = core.SigTime(start, stop, tA.sps)
    return yA, yB, xR, tI


# ========= 4) 路由式损失：主(A)等变，副(B)不变（商空间），显式 stop-grad 隔离 =========
def _evm_rms(y, x, eps=1e-8):
    num = jnp.mean(jnp.abs(y - x)**2)
    den = jnp.mean(jnp.abs(x)**2) + eps
    return num / den

def _loss_equivariant(y_pred, x_ref, beta_ce=0.5, tau_bce=1.5):
    """等变主损（不去相位）：Si-SNR + 0.1*EVM + β*Bit-BCE"""
    s = x_ref.reshape(-1); e = y_pred.reshape(-1)
    a = jnp.vdot(s, e) / (jnp.vdot(s, s) + 1e-8)
    s_hat = a * s
    err = e - s_hat
    snr_loss = -10.0 * jnp.log10((jnp.real(jnp.vdot(s_hat, s_hat))+1e-8) /
                                 (jnp.real(jnp.vdot(err, err))+1e-8))
    evm = _evm_rms(y_pred, x_ref)
    bce = _bit_bce_loss_16qam(y_pred, x_ref, tau=tau_bce)
    return snr_loss + 0.1 * evm + beta_ce * bce

def _loss_invariant(y_pred, x_ref, beta_ce=0.5, tau_bce=1.5):
    """相位不变副损：先仅去相位，再做 MSE + Bit-BCE"""
    yq = _align_phase_only(y_pred, x_ref)
    xq = _align_phase_only(x_ref,  x_ref)
    mse = jnp.mean(jnp.abs(yq - xq)**2)
    bce = _bit_bce_loss_16qam(yq, xq, tau=tau_bce)
    return 0.5 * mse + beta_ce * bce

def loss_fn(module, params, state, y, x, aux, const, sparams,
            lam_aux: float = 0.10, beta_ce: float = 0.5, tau_bce: float = 1.5):
    """
    L = L_main(A路) + lam_aux * L_aux(B路)
      - 主损： y_main = 0.5*(yA + stop_grad(yB))  只回 A
      - 副损： y_aux  = 0.5*(stop_grad(yA) + align_phase(yB)) 只回 B
    """
    params_net = util.dict_merge(params, sparams)
    (sigA, sigB), state_new = module.apply(
        {'params': params_net, 'aux_inputs': aux, 'const': const, **state},
        core.Signal(y)
    )

    # 对齐两路时间交集
    yA, yB, x_ref, _ = _crop_to_intersection(sigA, sigB, x)

    # —— 主损只更新 A —— 
    y_main = 0.5 * (yA + lax.stop_gradient(yB))
    L_main = _loss_equivariant(y_main, x_ref, beta_ce=beta_ce, tau_bce=tau_bce)

    # —— 副损只更新 B —— 
    yB_aln = _align_phase_only(yB, x_ref)
    y_aux  = 0.5 * (lax.stop_gradient(yA) + yB_aln)
    L_aux  = _loss_invariant(y_aux, x_ref, beta_ce=beta_ce, tau_bce=tau_bce)

    # 信息瓶颈（对两支轻约束，可选）
    ib = 0.5 * (jnp.mean(jnp.abs(yA)**2) + jnp.mean(jnp.abs(yB)**2))
    return (L_main + lam_aux * L_aux + 1e-4 * ib), state_new


# ========= 5) test：外部融合（默认均值；要换置信度收缩告诉我） =========
def test(model: Model,
         params: Dict,
         data: gdat.Input,
         eval_range: tuple=(300000, -20000),
         metric_fn=comm.qamqot):
    state, aux, const, sparams = model.initvar[1:]
    aux = core.dict_replace(aux, {'truth': data.x})
    if params is None:
        params = model.initvar[0]

    (sigA, sigB), _ = jax.jit(model.module.apply, backend='cpu')(
        {'params': util.dict_merge(params, sparams),
         'aux_inputs': aux, 'const': const, **state},
        core.Signal(data.y))

    # 对齐交集再融合
    yA, yB, x_ref, tI = _crop_to_intersection(sigA, sigB, data.x)
    z_val = 0.5 * (yA + yB)     # ★ 默认均值；可替换为置信度收缩
    z = core.Signal(z_val, tI)

    metric = metric_fn(
        z.val, x_ref,
        scale=np.sqrt(10), eval_range=eval_range
    )
    return metric, z


# ========= 6) equalize_dataset：同样外部融合 =========
def equalize_dataset(model_te, params, state_bundle, data):
    module_state, aux, const, sparams = state_bundle
    (sigA, sigB), _ = jax.jit(model_te.module.apply, backend='cpu')(
        {'params': util.dict_merge(params, sparams),
         'aux_inputs': aux, 'const': const, **module_state},
        core.Signal(data.y))

    yA, yB, x_ref, _ = _crop_to_intersection(sigA, sigB, np.asarray(data.x))
    z_eq  = np.asarray(0.5 * (yA + yB))        # equalized (fused)
    s_ref = np.asarray(x_ref)                  # 保持原尺度
    return z_eq, s_ref

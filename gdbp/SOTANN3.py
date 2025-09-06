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

def align_phase_scale(yhat: jnp.ndarray, y: jnp.ndarray, eps: float = 1e-8):
    """
    将预测 yhat 沿 U(1)×R+ 对齐到参考 y：
    min_{phi, alpha} || alpha * e^{-j phi} yhat - y ||^2
    返回对齐后的 yhat' 以及 (alpha, phi) 便于记录。
    """
    z = jnp.vdot(yhat, y)                          # <yhat, y>
    p = z / (jnp.abs(z) + eps)                     # 单位相位子（避免 atan2 奇异）
    yhp = yhat * jnp.conj(p)                       # 去相位：e^{-j phi} = conj(p)
    alpha = jnp.real(jnp.vdot(yhp, y)) / (jnp.real(jnp.vdot(yhp, yhp)) + eps)
    return alpha * yhp, alpha, jnp.angle(p)
  
CONST_16QAM = jnp.array([
    -3-3j, -3-1j, -3+3j, -3+1j,
    -1-3j, -1-1j, -1+3j, -1+1j,
     3-3j,  3-1j,  3+3j,  3+1j,
     1-3j,  1-1j,  1+3j,  1+1j
], dtype=jnp.complex64) / jnp.sqrt(10.)

# 2. CE-loss helper  (可 jit / vmap)
def _ce_loss_16qam(pred_sym: Array, true_sym: Array) -> Array:
    """
    pred_sym : [N] complex  — 网络输出符号
    true_sym : [N] complex  — 对齐后的发送符号
    return    : 标量 cross-entropy 损失
    """
    # logits =  –|y − s_k|²   (欧氏距离越小 → logit 越大)
    logits = -jnp.square(jnp.abs(pred_sym[..., None] - CONST_16QAM))   # [N,16]

    # 每个真实符号对应的 QAM 点下标
    label_idx = jnp.argmin(
        jnp.square(jnp.abs(true_sym[..., None] - CONST_16QAM)),
        axis=-1)                                                      # [N]

    # softmax-cross-entropy:
    log_prob = logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True)
    ce = -jnp.take_along_axis(log_prob, label_idx[..., None], axis=-1).squeeze(-1)

    return ce.mean()

_bits = (
    (0,0,0,0), (0,0,0,1), (0,0,1,0), (0,0,1,1),
    (0,1,0,0), (0,1,0,1), (0,1,1,0), (0,1,1,1),
    (1,0,0,0), (1,0,0,1), (1,0,1,0), (1,0,1,1),
    (1,1,0,0), (1,1,0,1), (1,1,1,0), (1,1,1,1)
)
BIT_MAP = jnp.array(_bits, dtype=jnp.float32)  # [16,4]
# 每 bit 的权重向量，顺序 = [b3(MSB), b2, b1, b0(LSB)]
BIT_WEIGHTS = jnp.array([1.2, 1.0, 1.0, 0.8], dtype=jnp.float32)

def _bit_bce_loss_16qam(pred_sym: Array, true_sym: Array) -> Array:
    logits  = -jnp.square(jnp.abs(pred_sym[..., None] - CONST_16QAM))   # [N,16]
    logp    = logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True) # [N,16]
    probs   = jnp.exp(logp)                                             # [N,16]

    # bit 概率 & 真值
    p1 = (probs @ BIT_MAP)                    # P(bit=1)
    p0 = 1.0 - p1

    idx  = jnp.argmin(jnp.square(jnp.abs(true_sym[..., None] - CONST_16QAM)), axis=-1)
    bits = BIT_MAP[idx]                       # 真值 bits

    bce  = -(bits * jnp.log(p1+1e-12) + (1.-bits) * jnp.log(p0+1e-12))  # [N,4]
    return (bce * BIT_WEIGHTS).mean()          # 加权平均
        
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
    bce = _bit_bce_loss_16qam(yq, xq)
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

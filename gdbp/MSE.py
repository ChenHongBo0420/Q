# Only mixed by the SNR loss vs Q
from jax import numpy as jnp, random, jit, value_and_grad, nn
import flax
from commplax import util, comm, comm2, cxopt, op, optim
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


def make_base_module(steps: int = 3,
                     dtaps: int = 261,
                     ntaps: int = 41,
                     rtaps: int = 61,
                     init_fn: tuple = (core.delta, core.gauss),
                     w0 = 0.,
                     mode: str = 'train'):
    '''
    make base module that derives DBP, FDBP, EDBP, GDBP depending on
    specific initialization method and trainable parameters defined
    by trainer.

    Args:
        steps: GDBP steps/layers
        dtaps: D-filter length
        ntaps: N-filter length
        rtaps: R-filter length
        init_fn: a tuple contains a pair of initializer for D-filter and N-filter
        mode: 'train' or 'test'

    Returns:
        A layer object
    '''

    _assert_taps(dtaps, ntaps, rtaps)

    d_init, n_init = init_fn

    if mode == 'train':
        # configure mimo to its training mode
        mimo_train = True
    elif mode == 'test':
        # mimo operates at training mode for the first 200000 symbols,
        # then switches to tracking mode afterwards
        mimo_train = cxopt.piecewise_constant([200000], [True, False])
    else:
        raise ValueError('invalid mode %s' % mode)
        
    base = layer.Serial(
        layer.FDBP(steps=steps,
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
        layer.vmap(layer.Conv1d)(name='RConv', taps=rtaps),  # vectorize column-wise Conv1D
        layer.MIMOAF(train=mimo_train))  # adaptive MIMO layer
        
    return base

def _assert_taps(dtaps, ntaps, rtaps, sps=2):
    """ we force odd taps to ease coding """
    assert dtaps % sps, f'dtaps must be odd number, got {dtaps} instead'
    assert ntaps % sps, f'ntaps must be odd number, got {ntaps} instead'
    assert rtaps % sps, f'rtaps must be odd number, got {rtaps} instead'


def fdbp_init(a: dict,
              xi: float = 1.1,
              steps: Optional[int] = None):
    """
    initializer for the base module (D/N filter profiles)
    """

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


def model_init(data: gdat.Input,
               base_conf: dict,
               sparams_flatkeys: list,
               n_symbols: int = 4000,
               sps: int = 2,
               name='Model'):
    """
    初始化模型：得到 module, (params, state, aux, const, sparams), overlaps
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


def l2_normalize(x, axis=None, epsilon=1e-12):
    square_sum = jnp.sum(jnp.square(x), axis=axis, keepdims=True)
    x_inv_norm = jnp.sqrt(jnp.maximum(square_sum, epsilon))
    return x / x_inv_norm


def _estimate_kmean(y, x, blk=16384):
    kap = []
    for i in range(0, len(x) - blk, blk):
        kap.append(np.abs(np.vdot(y[i:i + blk], x[i:i + blk]) /
                          np.vdot(x[i:i + blk], x[i:i + blk])))
    return float(np.mean(kap))


# ============================================
# 2) Batch-level 商空间对齐: 仅去除全局相位
# ============================================

def batch_phase_align(yhat: jnp.ndarray,
                      x_ref: jnp.ndarray,
                      eps: float = 1e-8) -> jnp.ndarray:
    """
    在一个 batch 上做 U(1) 商空间对齐：
      - 展平所有样本 / 维度，估计一个全局相位 φ*
      - 返回 e^{-j φ*} ⋅ yhat（与 x_ref 对齐）
    注意：φ* 本身不回传梯度（stop_gradient），
         避免网络学成退化“自己转相位抵消损失”的 trivial 解。
    """
    yh = yhat.reshape(-1)
    xr = x_ref.reshape(-1)

    zc = jnp.vdot(yh, xr)
    p = zc / (jnp.abs(zc) + eps)    # e^{j φ*}
    p = lax.stop_gradient(p)

    y_aligned = yhat * jnp.conj(p)  # e^{-j φ*} yhat
    return y_aligned


# ============================
# 3) Loss: SI-SNR in quotient space + EVM + tiny regularizer
# ============================

def loss_fn(module: layer.Layer,
            params: Dict,
            state: Dict,
            y: Array,
            x: Array,
            aux: Dict,
            const: Dict,
            sparams: Dict,
            beta_evm: float = 0.1,
            lambda_kl: float = 1e-4):
    """
    训练用 loss：
      - 先做 batch 商空间相位对齐；
      - 在去相位后的输出上算 SI-SNR（任务一致：类似 post-CPE 的 SNR）；
      - 加一点 EVM（更接近 Q²）作为 secondary term；
      - 再加非常轻微的能量正则（信息瓶颈风格）。

    这里故意不用 MSE 当主 loss，而是 SNR-style：
      loss_main = -SNR_dB + 0.1*EVM
    """

    params_net = util.dict_merge(params, sparams)
    z_out, state_new = module.apply(
        {'params': params_net, 'aux_inputs': aux, 'const': const, **state},
        core.Signal(y)
    )

    # 对齐时间区间
    x_ref = x[z_out.t.start:z_out.t.stop]   # [T,...]
    yhat = z_out.val                        # [T,...]

    # 1) 商空间相位对齐：只去除全局 φ*
    y_aligned = batch_phase_align(yhat, x_ref)

    # 2) SI-SNR（复数版）：在 y_aligned vs x_ref 上
    t = x_ref.reshape(-1)
    e = y_aligned.reshape(-1)

    num = jnp.vdot(t, e)
    den = jnp.vdot(t, t) + 1e-8
    alpha = num / den                      # 复数缩放
    s_hat = alpha * t
    err = e - s_hat

    s_energy = jnp.real(jnp.vdot(s_hat, s_hat)) + 1e-8
    n_energy = jnp.real(jnp.vdot(err, err)) + 1e-8
    snr_db = 10.0 * jnp.log10(s_energy / n_energy)

    # 我们最小化 loss → 用负 SNR
    snr_loss = -snr_db

    # 3) EVM（正规化到发射能量）
    evm_num = jnp.mean(jnp.abs(y_aligned - x_ref) ** 2)
    evm_den = jnp.mean(jnp.abs(x_ref) ** 2) + 1e-8
    evm = evm_num / evm_den

    # 4) 轻微能量正则（在原始 yhat 上，避免爆能量）
    kl_ib = 0.5 * jnp.mean(jnp.abs(yhat) ** 2)

    loss_main = snr_loss + beta_evm * evm
    total_loss = loss_main + lambda_kl * kl_ib

    return total_loss, state_new


# ============================
# 4) update_step / train
# ============================

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
    单步反向传播
    """
    params = opt.params_fn(opt_state)
    (loss, module_state), grads = value_and_grad(
        loss_fn, argnums=1, has_aux=True
    )(module, params, module_state, y, x, aux, const, sparams)

    opt_state = opt.update_fn(i, grads, opt_state)
    return loss, opt_state, module_state


def get_train_batch(ds: gdat.Input,
                    batchsize: int,
                    overlaps: int,
                    sps: int = 2):
    """
    生成带 overlap 的训练 batch
    """
    flen = batchsize + overlaps
    fstep = batchsize
    ds_y = op.frame_gen(ds.y, flen * sps, fstep * sps)
    ds_x = op.frame_gen(ds.x, flen, fstep)
    n_batches = op.frame_shape(ds.x.shape, flen, fstep)[0]
    return n_batches, zip(ds_y, ds_x)


def train(model: Model,
          data: gdat.Input,
          batch_size: int = 500,
          n_iter=None,
          opt: optim.Optimizer = optim.adam(
              optim.piecewise_constant([500, 1000], [1e-4, 1e-5, 1e-6])
          ),
          diag: Optional[dict] = None):
    """
    训练 1 epoch，yield (loss, params, module_state) —— 保持原接口不变。

    额外功能（可选）：
    传入 diag=dict(...) 后，会在训练过程中每隔 eval_every 迭代跑一次 test()
    并把诊断指标记录到 train.diag_logs（list[dict]）里。

    diag 支持的键（都可选，没给就用默认）：
      - model_eval : Model   （必填，想算诊断就要给 test 用的 model）
      - data_eval  : gdat.Input（必填）
      - eval_every : int，默认 200
      - eval_range : tuple，默认 (300000, -20000)（同 test()）
      - L          : int，默认 16
      - d4_dims    : int，默认 2
      - pilot_frac : float，默认 0.0
      - use_oracle_noise : bool，默认 True
      - use_elliptical_llr : bool，默认 True
      - temp_grid  : tuple，默认 (0.75, 1.0, 1.25)
      - bitwidth   : int，默认 6
      - decoder    : decoder，默认 None
      - cpe_block_len : int，默认 2048 （用于 phi_blk 指标）
      - callback   : callable(rec: dict) -> None（每次记录后回调）
    """

    # -------- 0) 准备诊断配置（不影响原训练流程）--------
    train.diag_logs = []  # 外部训练完可直接访问：gb.train.diag_logs
    if diag is not None:
        if ('model_eval' not in diag) or ('data_eval' not in diag):
            raise ValueError("diag 开启时必须提供 diag['model_eval'] 和 diag['data_eval']")

        model_eval = diag['model_eval']
        data_eval = diag['data_eval']
        eval_every = int(diag.get('eval_every', 200))
        cpe_block_len = int(diag.get('cpe_block_len', 2048))
        cb = diag.get('callback', None)

        # test() 的参数（尽量跟你现在脚本一致）
        test_kwargs = dict(
            eval_range=diag.get('eval_range', (300000, -20000)),
            L=int(diag.get('L', 16)),
            d4_dims=int(diag.get('d4_dims', 2)),
            pilot_frac=float(diag.get('pilot_frac', 0.0)),
            use_oracle_noise=bool(diag.get('use_oracle_noise', True)),
            use_elliptical_llr=bool(diag.get('use_elliptical_llr', True)),
            temp_grid=diag.get('temp_grid', (0.75, 1.0, 1.25)),
            bitwidth=int(diag.get('bitwidth', 6)),
            decoder=diag.get('decoder', None),
            return_artifacts=False
        )

        # 训练内用的小工具：从 test 输出的 z / data_eval 估计块级残余相位
        def _phi_blk_metrics_from_z(z_sig, data_eval, L, eval_range, block_len):
            # 复制 test() 中的 canonical scaling + eval_range 裁剪口径 :contentReference[oaicite:2]{index=2}
            x_ref = _np.asarray(data_eval.x[z_sig.t.start:z_sig.t.stop])
            scale = comm2.qamscale(L) if L is not None else _np.sqrt(10.0)
            y_eq = _np.asarray(z_sig.val) * scale
            x_ref = x_ref * scale

            s0, s1 = eval_range
            s1 = y_eq.shape[0] + s1 if s1 <= 0 else s1
            y_eq = y_eq[s0:s1]
            x_ref = x_ref[s0:s1]

            T = min(y_eq.shape[0], x_ref.shape[0])
            y_eq = y_eq[:T]
            x_ref = x_ref[:T]

            n_blk = T // block_len
            if n_blk < 2:
                return _np.nan, _np.nan, int(n_blk)

            phis = []
            for b in range(n_blk):
                ys = y_eq[b * block_len:(b + 1) * block_len].reshape(-1)
                xs = x_ref[b * block_len:(b + 1) * block_len].reshape(-1)
                zc = _np.vdot(ys, xs)
                phis.append(_np.angle(zc))
            phis = _np.unwrap(_np.asarray(phis))

            t = _np.arange(len(phis))
            a, b0 = _np.polyfit(t, phis, 1)
            res = phis - (a * t + b0)
            rms = float(_np.sqrt(_np.mean(res ** 2)))
            return rms, float(a), int(n_blk)

    # -------- 1) 原 train() 初始化（保持一致）--------
    params, module_state, aux, const, sparams = model.initvar
    opt_state = opt.init_fn(params)

    n_batch, batch_gen = get_train_batch(data, batch_size, model.overlaps)
    n_iter = n_batch if n_iter is None else min(n_iter, n_batch)

    # -------- 2) 训练主循环（yield 行为保持一致）--------
    for i, (y, x) in tqdm(enumerate(batch_gen),
                          total=n_iter, desc='training', leave=False):
        if i >= n_iter:
            break

        aux = core.dict_replace(aux, {'truth': x})
        loss, opt_state, module_state = update_step(
            model.module, opt, i, opt_state,
            module_state, y, x, aux, const, sparams
        )
        cur_params = opt.params_fn(opt_state)

        # -------- 3) 可选：周期性诊断（不改变 yield 输出）--------
        if diag is not None and (i % eval_every == 0):
            res, z = test(model_eval, cur_params, data_eval, **test_kwargs)
            sd = res.get("SD", {})
            hd = res.get("HD", None)

            # 从 test() 里直接取（你已经算好的）MSE_raw / MSE_qspace :contentReference[oaicite:3]{index=3}
            mse_raw = float(sd.get("MSE_raw", _np.nan))
            mse_qspace = float(sd.get("MSE_qspace", _np.nan))
            ngmi_4d = float(sd.get("NGMI_4D", _np.nan))
            gmi_4d = float(sd.get("GMI_bits_per_4D", _np.nan))
            air_4d = float(sd.get("AIR_bits_per_4D", _np.nan))

            # 额外：残余相位时变（块级）
            phi_rms, phi_slope, n_blk = _phi_blk_metrics_from_z(
                z_sig=z,
                data_eval=data_eval,
                L=test_kwargs["L"],
                eval_range=test_kwargs["eval_range"],
                block_len=cpe_block_len
            )

            # HD 总指标（可选）
            qsq = _np.nan
            ber = _np.nan
            try:
                if hd is not None:
                    qsq = float(hd.loc["total", "QSq"])
                    ber = float(hd.loc["total", "BER"])
            except Exception:
                pass

            rec = dict(
                it=int(i),
                loss=float(_np.asarray(loss)),
                Q_dB=qsq,
                BER=ber,
                NGMI_4D=ngmi_4d,
                GMI_b_per_4D=gmi_4d,
                AIR_b_per_4D=air_4d,
                MSE_raw=mse_raw,
                MSE_qspace=mse_qspace,
                phi_blk_residual_rms_rad=phi_rms,
                phi_blk_slope_rad_per_block=phi_slope,
                cpe_n_blocks=n_blk,
                cpe_block_len=int(cpe_block_len),
            )
            train.diag_logs.append(rec)
            if cb is not None:
                cb(rec)

        # 仍然按原来的格式 yield —— 外部脚本无需改任何一行 :contentReference[oaicite:4]{index=4}
        yield loss, cur_params, module_state


# ============================
# 5) test：HD + SD 统一评估 + MSE_raw/MSE_qspace
# ============================

def test(model: Model,
         params: Dict,
         data: gdat.Input,
         eval_range: tuple = (300000, -20000),
         L: int = 16,
         d4_dims: int = 2,
         pilot_frac: float = 0.0,
         use_oracle_noise: bool = True,
         use_elliptical_llr: bool = True,
         temp_grid: tuple = (0.75, 1.0, 1.25),
         bitwidth: int = 6,
         decoder=None,
         return_artifacts: bool = False):
    """
    前向 + HD/SD 统一评估
      res = {
        'HD': DataFrame(only 'total' 行, 含 BER / Q / SNR),
        'SD': dict(含 GMI / NGMI / AIR / temp / alpha 等,
                   以及附加 MSE_raw / MSE_qspace)
      }, z = 等化后的 Signal 对象
    """

    state, aux, const, sparams = model.initvar[1:]
    aux = core.dict_replace(aux, {'truth': data.x})
    if params is None:
        params = model.initvar[0]

    z, _ = jit(model.module.apply, backend='cpu')({
        'params': util.dict_merge(params, sparams),
        'aux_inputs': aux, 'const': const, **state
    }, core.Signal(data.y))

    # canonical 缩放，让 x_ref 落在标准 QAM 网格
    x_ref = data.x[z.t.start:z.t.stop]
    scale = comm2.qamscale(L) if L is not None else np.sqrt(10.0)
    y_eq = z.val * scale
    x_ref = x_ref * scale

    # eval_range 裁剪
    s0, s1 = eval_range
    s1 = y_eq.shape[0] + s1 if s1 <= 0 else s1
    y_eq = y_eq[s0:s1]
    x_ref = x_ref[s0:s1]

    # flatten 复维
    y_1d = y_eq.reshape(-1)
    x_1d = x_ref.reshape(-1)

    # ==== 新增：MSE_raw / MSE_qspace 作为诊断量 ====
    # raw MSE（含残余相位）
    mse_raw = float(jnp.mean(jnp.abs(y_eq - x_ref) ** 2))

    # 商空间 MSE（去掉 batch 全局相位）
    y_q = batch_phase_align(jnp.asarray(y_eq), jnp.asarray(x_ref))
    mse_qspace = float(jnp.mean(jnp.abs(y_q - jnp.asarray(x_ref)) ** 2))
    # ===========================================

    # SD 评估：估噪 → LLR → 温度扫描 → (可选)软解码 → GMI
    sd_kwargs = dict(
        use_oracle_noise=use_oracle_noise,
        elliptical_llr=use_elliptical_llr,
        temp_grid=temp_grid,
        bitwidth=bitwidth,
        return_artifacts=return_artifacts
    )
    res = comm2.evaluate_hd_and_sd(
        y_1d, x_1d,
        L=L if L is not None else 16,
        decoder=decoder,
        sd_kwargs=sd_kwargs
    )

    # 4D 聚合指标
    m_per_dim = int(np.log2(L if L is not None else 16))
    gmi_dim = float(res['SD']['GMI_bits_per_dim'])
    gmi_4d = gmi_dim * d4_dims
    ngmi_4d = gmi_4d / (m_per_dim * d4_dims)
    air_4d = gmi_4d * (1.0 - pilot_frac)

    res['SD'].update({
        'GMI_bits_per_4D': gmi_4d,
        'NGMI_4D': ngmi_4d,
        'AIR_bits_per_4D': air_4d,
        'pilot_frac': pilot_frac,
        'd4_dims': d4_dims,
        # 把两种 MSE 一起挂在 SD 里方便你后处理
        'MSE_raw': mse_raw,
        'MSE_qspace': mse_qspace,
    })

    return res, z


# ============================
# 6) equalize_dataset（保持不变）
# ============================

def equalize_dataset(model_te, params, state_bundle, data):
    module_state, aux, const, sparams = state_bundle
    z, _ = jax.jit(model_te.module.apply, backend='cpu')(
        {'params': util.dict_merge(params, sparams),
         'aux_inputs': aux, 'const': const, **module_state},
        core.Signal(data.y))

    start, stop = z.t.start, z.t.stop
    z_eq = np.asarray(z.val[:, 0])       # equalized
    s_ref = np.asarray(data.x)[start:stop, 0]  # 保持原尺度
    return z_eq, s_ref


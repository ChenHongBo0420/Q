# MSE problem for Q vs Loss
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
                        w0=0.,
                        mode: str = 'train',
                        # ===== NEW: FOE/CPE loop knobs =====
                        framesize: int = 100,
                        foe_strength: float = 1.0,
                        foekwargs: dict | None = None,
                        # ===== NEW: DDLMS step sizes =====
                        mu_h: float | None = None,
                        mu_f: float | None = None,
                        mu_s: float | None = None,
                        mu_b: float | None = None,
                        # allow user override additional mimokwargs
                        mimokwargs: dict | None = None):
    """
    与你原 make_base_module 结构完全一致，只是：
      1) MIMOFOEAf 支持 framesize / foe_strength / foekwargs
      2) MIMOAF(DDLMS) 支持 mu_h/mu_f/... 并映射到 ddlms 的 lr_w/lr_f/lr_s/lr_b
    """

    # ---- keep your original train/test policy ----
    if mode == 'train':
        mimo_train = True
    elif mode == 'test':
        mimo_train = cxopt.piecewise_constant([200000], [True, False])
    else:
        raise ValueError(f'invalid mode {mode}')

    d_init, n_init = init_fn
    if foekwargs is None:
        foekwargs = {}

    # ---- build mimokwargs for DDLMS ----
    # DDLMS 的真实参数名：lr_w/lr_f/lr_s/lr_b（不是 lr）
    mk = {} if mimokwargs is None else dict(mimokwargs)
    if mu_h is not None:
        mk["lr_w"] = float(mu_h)
    if mu_f is not None:
        mk["lr_f"] = float(mu_f)
    if mu_s is not None:
        mk["lr_s"] = float(mu_s)
    if mu_b is not None:
        mk["lr_b"] = float(mu_b)

    base = layer.Serial(
        layer.FDBP(steps=steps,
                   dtaps=dtaps,
                   ntaps=ntaps,
                   d_init=d_init,
                   n_init=n_init),
        layer.BatchPowerNorm(mode=mode),

        # ===== FOE/CPE + MIMO-for-FOE =====
        # commplax.module.core.mimofoeaf 的签名里有 framesize/foe_strength/foekwargs :contentReference[oaicite:3]{index=3}
        layer.MIMOFOEAf(name='FOEAf',
                        w0=w0,
                        train=mimo_train,
                        preslicer=core.conv1d_slicer(rtaps),
                        foekwargs=foekwargs,
                        framesize=int(framesize),
                        foe_strength=float(foe_strength)),

        layer.vmap(layer.Conv1d)(name='RConv', taps=rtaps),

        # ===== final adaptive MIMO (DDLMS) =====
        # mimoaf 会把 mimokwargs 喂给 mimofn(train=..., **mimokwargs) :contentReference[oaicite:4]{index=4}
        layer.MIMOAF(train=mimo_train, mimokwargs=mk),
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


def model_init(data,
                  base_conf: dict,
                  sparams_flatkeys: list,
                  n_symbols: int = 4000,
                  sps: int = 2,
                  name='Model'):
    """
    完全复刻你原 model_init 的结构，只是 mod = make_base_module_v2(...)
    """
    # ---- core change: call v2 ----
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
    mse_loss = jnp.mean(jnp.abs(z_original.val - aligned_x) ** 2)

    return mse_loss, updated_state
              
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
          n_iter=None,
          opt: optim.Optimizer = optim.adam(
              optim.piecewise_constant([500, 1000], [1e-4, 1e-5, 1e-6])
          ),
          diag: Optional[dict] = None):
    """
    训练 1 epoch，yield (loss, params, module_state) —— 保持原接口不变。

    diag 开启后：每隔 eval_every 跑一次 test()，只记录 Q/BER + MSE_raw/MSE_qspace
    + 相位块级稳定性指标（phi_rms / std(dphi_blk) / slip_count），以及可选的 MIMO taps 漂移。
    """

    # -------- 0) 准备诊断配置（不影响原训练流程）--------
    train.diag_logs = []
    if diag is not None:
        if ('model_eval' not in diag) or ('data_eval' not in diag):
            raise ValueError("diag 开启时必须提供 diag['model_eval'] 和 diag['data_eval']")

        model_eval = diag['model_eval']
        data_eval = diag['data_eval']
        eval_every = int(diag.get('eval_every', 200))
        cpe_block_len = int(diag.get('cpe_block_len', 2048))
        cb = diag.get('callback', None)

        # 可选：提取 MIMO taps 的函数（你自己提供，返回一个 ndarray/DeviceArray）
        # 例子：diag['taps_getter'] = lambda params, module_state: params['mimo']['W']
        taps_getter = diag.get('taps_getter', None)

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

        def _phi_blk_metrics_from_z(z_sig, data_eval, L, eval_range, block_len):
            """
            返回：
              phi_rms         : 去线性趋势后的 block-phase 残余 RMS
              phi_slope       : block-phase 线性斜率（频偏/漂移 proxy）
              dphi_std        : std(Δphi_blk) —— block-to-block 更新量 RMS（稳定性）
              slip_count      : slip 计数（以 raw 相位差跨越 π 作为 wrap/slip proxy）
              n_blk           : block 数
            """
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
                return _np.nan, _np.nan, _np.nan, 0, int(n_blk)

            # 1) 每块一个“原始相位”phi_raw[b] = angle(<y_b, x_b>)
            phi_raw = _np.empty((n_blk,), dtype=_np.float64)
            for b in range(n_blk):
                ys = y_eq[b * block_len:(b + 1) * block_len].reshape(-1)
                xs = x_ref[b * block_len:(b + 1) * block_len].reshape(-1)
                zc = _np.vdot(ys, xs)
                phi_raw[b] = _np.angle(zc)

            # 2) slip proxy：看 raw 相位差是否跨越 π（wrap）
            dphi_raw = _np.diff(phi_raw)
            slip_count = int(_np.sum(_np.abs(dphi_raw) > _np.pi))

            # 3) unwrap 后做线性拟合，得到 residual RMS
            phi = _np.unwrap(phi_raw)
            t = _np.arange(n_blk)
            a, b0 = _np.polyfit(t, phi, 1)
            res = phi - (a * t + b0)
            phi_rms = float(_np.sqrt(_np.mean(res ** 2)))
            phi_slope = float(a)

            # 4) block-to-block 更新量 RMS：std(Δphi_blk)（用 unwrap 后的差分更合理）
            dphi = _np.diff(phi)
            dphi_std = float(_np.std(dphi))

            return phi_rms, phi_slope, dphi_std, slip_count, int(n_blk)

        # 用于计算 ||W|| 和 ||ΔW|| 的缓存
        _prev_W = None

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

        # -------- 3) 可选：周期性诊断（只记 Q/BER + 跟踪稳定性）--------
        if diag is not None and (i % eval_every == 0):
            res, z = test(model_eval, cur_params, data_eval, **test_kwargs)
            sd = res.get("SD", {})
            hd = res.get("HD", None)

            # test() 里已有的两种 MSE 诊断
            mse_raw = float(sd.get("MSE_raw", _np.nan))
            mse_qspace = float(sd.get("MSE_qspace", _np.nan))

            # 相位块级稳定性指标
            phi_rms, phi_slope, dphi_std, slip_count, n_blk = _phi_blk_metrics_from_z(
                z_sig=z,
                data_eval=data_eval,
                L=test_kwargs["L"],
                eval_range=test_kwargs["eval_range"],
                block_len=cpe_block_len
            )

            # 只取 HD 的 Q/BER
            qsq = _np.nan
            ber = _np.nan
            try:
                if hd is not None:
                    qsq = float(hd.loc["total", "QSq"])
                    ber = float(hd.loc["total", "BER"])
            except Exception:
                pass

            # 可选：MIMO taps 漂移（需要你提供 taps_getter）
            W_norm = _np.nan
            dW_norm = _np.nan
            if taps_getter is not None:
                try:
                    W = taps_getter(cur_params, module_state)
                    W = _np.asarray(W)
                    W_norm = float(_np.linalg.norm(W.reshape(-1)))
                    if _prev_W is not None:
                        dW_norm = float(_np.linalg.norm((W - _prev_W).reshape(-1)))
                    _prev_W = W
                except Exception:
                    # 不让诊断影响训练
                    pass

            rec = dict(
                it=int(i),
                loss=float(_np.asarray(loss)),

                # 最终判决性能（你要的主线）
                Q_dB=qsq,
                BER=ber,

                # 几何误差诊断
                MSE_raw=mse_raw,
                MSE_qspace=mse_qspace,

                # 跟踪稳定性诊断（你点名要的三项）
                phi_blk_residual_rms_rad=phi_rms,
                phi_blk_slope_rad_per_block=phi_slope,
                phi_blk_dphi_std_rad=dphi_std,      # std(Δphi_blk)
                phi_blk_slip_count=slip_count,      # wrap/slip proxy

                cpe_n_blocks=n_blk,
                cpe_block_len=int(cpe_block_len),

                # 可选：MIMO taps 漂移（如果 taps_getter 给了）
                mimo_W_norm=W_norm,
                mimo_dW_norm=dW_norm,
            )
            train.diag_logs.append(rec)
            if cb is not None:
                cb(rec)

        # 仍然按原来的格式 yield
        yield loss, cur_params, module_state


 

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
         return_artifacts: bool = False,
         # ===== NEW (optional) =====
         return_state: bool = False,
         inject_action_metrics: bool = True):
    """
    返回：
      - 默认：res, z
      - return_state=True：res, z, updated_state

    兼容增强：
      - 若传入 params 是空 dict/空 FrozenDict：自动回退到 model.initvar[0]，避免 ScopeCollectionNotFound
      - 尝试从 updated_state['af_state'][...]['framefoeaf'] 的 tuple 里抽“动作强度”并写入 res['SD']
    """

    import numpy as _np

    # --------------------------
    # 0) helpers
    # --------------------------
    def _is_empty_params(p):
        # 兼容 dict / FrozenDict：只要“看起来没有任何 key”，就认为是空 params
        try:
            if p is None:
                return True
            # FrozenDict/dict 都支持 keys()
            return len(list(p.keys())) == 0
        except Exception:
            # 万一是别的 pytree 类型，尽量别误伤
            return False

    def _walk_collect(obj, prefix=()):
        """把 pytree 里所有叶子收集成 (path_tuple, leaf) 列表（轻量，不依赖 flax.traverse_util）"""
        out = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                out.extend(_walk_collect(v, prefix + (str(k),)))
        else:
            # flax FrozenDict 也当 dict 处理
            try:
                import flax
                if isinstance(obj, flax.core.FrozenDict):
                    for k, v in dict(obj).items():
                        out.extend(_walk_collect(v, prefix + (str(k),)))
                else:
                    out.append((prefix, obj))
            except Exception:
                out.append((prefix, obj))
        return out

    def _extract_framefoeaf_tuple(state_tree):
        """
        在 updated_state 里找 path 末尾包含 ('af_state', ..., 'framefoeaf') 的那个叶子。
        你 core.mimofoeaf 里是：
            scope.variable('af_state', 'framefoeaf', ...)
        """
        leaves = _walk_collect(state_tree, prefix=())
        cand = []
        for path, leaf in leaves:
            p = "/".join(path).lower()
            if ("af_state" in p) and ("framefoeaf" in p):
                cand.append((path, leaf))
        if not cand:
            return None
        # 如果有多个，优先选最深的（更具体的那条）
        cand.sort(key=lambda x: len(x[0]), reverse=True)
        return cand[0][1]

    def _safe_float(x):
        try:
            return float(_np.asarray(x))
        except Exception:
            return _np.nan

    # --------------------------
    # 1) forward (get z + updated_state)
    # --------------------------
    state, aux, const, sparams = model.initvar[1:]
    aux = core.dict_replace(aux, {'truth': data.x})

    # 关键：空 params -> fallback，修你遇到的 ScopeCollectionNotFound
    if _is_empty_params(params):
        params = model.initvar[0]

    z, updated_state = jit(model.module.apply, backend='cpu')({
        'params': util.dict_merge(params, sparams),
        'aux_inputs': aux, 'const': const, **state
    }, core.Signal(data.y))

    # --------------------------
    # 2) align + canonical scale
    # --------------------------
    x_ref = data.x[z.t.start:z.t.stop]
    scale = comm2.qamscale(L) if L is not None else _np.sqrt(10.0)
    y_eq  = _np.asarray(z.val) * scale
    x_ref = _np.asarray(x_ref) * scale

    # eval_range slice
    s0, s1 = eval_range
    s1 = y_eq.shape[0] + s1 if s1 <= 0 else s1
    s0 = max(0, s0)
    s1 = min(y_eq.shape[0], s1)
    y_eq = y_eq[s0:s1]
    x_ref = x_ref[s0:s1]

    # flatten to 1D for HD/SD
    y_1d = y_eq.reshape(-1)
    x_1d = x_ref.reshape(-1)

    # --------------------------
    # 3) HD + SD
    # --------------------------
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

    # 4D aggregation
    m_per_dim = int(_np.log2(L if L is not None else 16))
    gmi_dim   = float(res['SD'].get('GMI_bits_per_dim', _np.nan))
    gmi_4d    = gmi_dim * d4_dims
    ngmi_4d   = gmi_4d / (m_per_dim * d4_dims)
    air_4d    = gmi_4d * (1.0 - pilot_frac)

    res['SD'].update({
        'GMI_bits_per_4D': gmi_4d,
        'NGMI_4D': ngmi_4d,
        'AIR_bits_per_4D': air_4d,
        'pilot_frac': pilot_frac,
        'd4_dims': d4_dims
    })

    # 如果 evaluate_hd_and_sd 没给 MSE，我们补上（按你 test 口径：scale+eval_range 后的 y_eq/x_ref）
    if ('MSE_raw' not in res['SD']) or ('MSE_qspace' not in res['SD']):
        eps = 1e-8
        mse_raw = float(_np.mean(_np.abs(y_1d - x_1d) ** 2))
        zc = _np.vdot(y_1d, x_1d)
        p = zc / (abs(zc) + eps)
        y_aligned = y_1d * _np.conj(p)
        mse_qsp = float(_np.mean(_np.abs(y_aligned - x_1d) ** 2))
        res['SD']['MSE_raw'] = mse_raw
        res['SD']['MSE_qspace'] = mse_qsp

    # --------------------------
    # 4) (NEW) inject “action strength” from updated_state
    # --------------------------
    if inject_action_metrics:
        # 你在 core.mimofoeaf 里如果把动作量写进了 framefoeaf 的 tuple，
        # 这里就能直接取出来并塞进 res['SD']。
        t = _extract_framefoeaf_tuple(updated_state)

        # 兼容两种 state 形态：
        #   old: (phi_last, af_step, af_stats)
        #   new: (phi_last, af_step, af_stats, act_rms, act_std, act_mean_abs, ...)
        if isinstance(t, tuple) and (len(t) >= 3):
            res['SD']['FOE_phi_last_rad'] = _safe_float(t[0])
            res['SD']['FOE_af_step'] = _safe_float(t[1])

            # 如果你 core 里加了动作强度（推荐你用：act_rms / act_mean_abs）
            if len(t) >= 4:
                res['SD']['FOE_action_rms'] = _safe_float(t[3])
            if len(t) >= 5:
                res['SD']['FOE_action_std'] = _safe_float(t[4])
            if len(t) >= 6:
                res['SD']['FOE_action_mean_abs'] = _safe_float(t[5])

            # 你要是还塞了别的统计量，也可以继续往后加
            # if len(t) >= 7: res['SD']['FOE_xxx'] = _safe_float(t[6])

    if return_state:
        return res, z, updated_state
    return res, z




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

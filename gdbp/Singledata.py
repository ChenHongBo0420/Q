# === gdbp/Singledata.py ============================================
import zarr, numpy as np
from itertools import chain
from scipy.signal import correlate
from commplax import comm
from collections import namedtuple
from tqdm.auto import tqdm
import labptptm1

Input = namedtuple('DataInput', ['y', 'x', 'w0', 'a'])

# -------------------------------------------------------------------
# metadata helpers
# -------------------------------------------------------------------
def _parent_group(grp):
    if not grp.path:          # root
        return None
    par = '/'.join(grp.path.split('/')[:-1])
    return zarr.open_group(store=grp.store, path=par, mode='r')

def _get_meta(grp, *names, default=None):
    targets = {n.lower() for n in chain.from_iterable(
               n.split('|') for n in names)}
    g = grp
    while g is not None:
        for k, v in g.attrs.items():
            if k.lower() in targets:
                return v
        g = _parent_group(g)
    return default

# -------------------------------------------------------------------
# align two polarizations independently
# -------------------------------------------------------------------
def _estimate_lag(y_1d, x_1d):
    """整数符号级互相关提前量；recv 早 ⇒ lag<0"""
    return (np.argmax(np.abs(correlate(y_1d, x_1d, 'full')))
            - (len(x_1d) - 1))

def _align_and_rotate(y, x, sps, chk=12_000):
    """
    y: (Nsamp,2)   x: (Nsym,2)   -> 同步裁剪 + 常量相位补偿
    返回 y, x, [lag0, lag1], phi
    """
    N  = min(chk, len(x))
    lag0 = _estimate_lag(y[:N*sps:sps, 0], x[:N, 0])
    lag1 = _estimate_lag(y[:N*sps:sps, 1], x[:N, 1])
    lag_min = min(lag0, lag1)

    # 裁剪各自偏振
    offs0 = (lag0 - lag_min) * sps
    offs1 = (lag1 - lag_min) * sps
    y0 = y[offs0:, 0]
    y1 = y[offs1:, 1]
    L  = min(len(y0), len(y1))
    y  = np.stack([y0[:L], y1[:L]], axis=1)
    x  = x[-lag_min : -lag_min + L//sps]

    # 常量相位（两路一起平均）
    phi = np.angle(np.mean(
        x[:N].reshape(-1) * np.conj(y[:N*sps:sps].reshape(-1))
    ))
    y *= np.exp(1j * phi)
    return y, x, [lag0, lag1], phi

# -------------------------------------------------------------------
# core loader
# -------------------------------------------------------------------
def _loader(dg, n_sym, lp_dbm):
    br   = _get_meta(dg, 'baudrate|symbolrate')
    fs   = _get_meta(dg, 'samplerate|sample_rate|fs')
    dist = _get_meta(dg, 'distance', default=815e3)   # m
    spans= _get_meta(dg, 'spans|nspans', default=10)  # 段数
    if br is None or fs is None:
        raise KeyError("missing 'baudrate' or 'samplerate'")

    sps  = int(round(fs / br))
    mod  = _get_meta(dg, 'modformat', default='16QAM')
    beta = _get_meta(dg, 'rolloff',  default=0.2)

    # 1) 读取足够长的波形
    y = dg['recv'][: (n_sym + 4_000) * sps]
    x = dg['sent'][: n_sym]

    # 2) 正向归一化
    y = y - np.mean(y, axis=0)
    y = comm.normpower(y, real=True) / np.sqrt(2)
    x = x / comm.qamscale(mod)

    # 3) 对齐 + 相位
    y, x, lag_list, phi = _align_and_rotate(y, x, sps)

    # 4) 打包
    a = dict(
        samplerate = fs,
        symbolrate = br,
        sps        = sps,
        distance   = dist,
        spans      = spans,
        rolloff    = beta,
        modformat  = mod,
        lpdbm      = lp_dbm,
        lag_list   = lag_list,          # [lag_pol0, lag_pol1]
        phi_rad    = float(phi),
    )
    return Input(y=y, x=x, w0=0.0, a=a)

# -------------------------------------------------------------------
# public API
# -------------------------------------------------------------------
def load(mod, lp_dbm, rep, n_symbols=1_500_000):
    grps, _ = labptptm1.select(mod, lp_dbm, rep)
    return [ _loader(g, n_symbols, lp_dbm) for g in
             tqdm(grps, desc='loading LabPtPtm1', leave=False) ]
# ===================================================================

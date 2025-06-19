# === Singledata.py ================================================
import zarr, numpy as np
from itertools import chain
from scipy.signal import correlate
from commplax import comm
from collections import namedtuple
from tqdm.auto import tqdm
import labptptm1

# ---------------- container ----------------
Input = namedtuple('DataInput', ['y', 'x', 'w0', 'a'])

# ---------------- helper: Zarr 元数据递归 ----------------
def _parent_group(grp):
    if not grp.path:            # root
        return None
    p = '/'.join(grp.path.split('/')[:-1])
    return zarr.open_group(store=grp.store, path=p, mode='r')

def _get_meta(grp, *names):
    """大小写无关 + 多别名查询；一直爬到根组"""
    targets = {n.lower() for n in chain.from_iterable(n.split('|') for n in names)}
    g = grp
    while g is not None:
        for k, v in g.attrs.items():
            if k.lower() in targets:
                return v
        g = _parent_group(g)
    return None

# ---------------- helper: lag & 相位 ----------------
def _estimate_lag(y_ds_1d, x_ref):
    """互相关估整数符号延迟；recv 早 ⇒ lag<0。"""
    return (np.argmax(np.abs(correlate(y_ds_1d, x_ref, mode='full')))
            - (len(x_ref) - 1))

def _align_and_rotate(y, x, sps, max_sym=10_000):
    n_chk = min(max_sym, len(x))

    # --- 1) 互相关估 lag（第 0 偏振） ---
    y_ds = y[:n_chk*sps:sps, 0] if y.ndim == 2 else y[:n_chk*sps:sps]
    x_r  = x[:n_chk, 0]         if x.ndim == 2 else x[:n_chk]
    lag  = (np.argmax(np.abs(correlate(y_ds, x_r, mode='full')))
            - (len(x_r) - 1))

    # --- 2) 根据 lag 裁剪 ---
    if lag < 0:          # recv 早
        y = y[-lag*sps:]
        x = x[:len(y)//sps]
    elif lag > 0:        # recv 晚
        x = x[lag:]
        y = y[:len(x)*sps]

    # --- 3) 常量相位（仍用第 0 偏振） ---
    y_ds = y[:n_chk*sps:sps, 0] if y.ndim == 2 else y[:n_chk*sps:sps]
    x_r  = x[:n_chk, 0]         if x.ndim == 2 else x[:n_chk]
    phi  = np.angle(np.mean(x_r * np.conj(y_ds)))

    # 同一个 φ 旋转全部偏振
    y *= np.exp(1j*phi)
    return y, x, lag, phi


# ---------------- 核心 loader ----------------
def _loader(dat_grp, n_symbols, lp_dbm):
    # 1) 读基本元数据
    baudrate   = _get_meta(dat_grp, 'baudrate|symbolrate')
    samplerate = _get_meta(dat_grp, 'samplerate|sample_rate|fs')
    if baudrate is None or samplerate is None:
        raise KeyError("Missing 'baudrate' or 'samplerate' in attrs tree.")
    sps = int(round(samplerate / baudrate))

    distance = _get_meta(dat_grp, 'distance')
    spans    = _get_meta(dat_grp, 'spans')
    modfmt   = _get_meta(dat_grp, 'modformat') or '16QAM'
    cd       = _get_meta(dat_grp, 'cd', 'dispersion') or 17e-6
    fo_hz    = _get_meta(dat_grp, 'fo', 'freq_offset') or 0.0

    # 2) 读取波形 (保证足够长，便于裁剪)
    n_samples = int((n_symbols + 4_000) * sps)   # +4k 留冗余
    y = dat_grp['recv'][:n_samples]
    x = dat_grp['sent'][:n_symbols]

    # 3) CFO 粗补偿
    if fo_hz:
        t = np.arange(len(y))
        y *= np.exp(-1j * 2*np.pi * fo_hz * t / samplerate)

    # 4) 去均值 + 归一化 + sent 缩放
    y -= np.mean(y, axis=0)
    y  = comm.normpower(y, real=True) / np.sqrt(2)
    x  = x / comm.qamscale(modfmt)

    # 5) 对齐 + 相位
    y, x, lag_sym, phi = _align_and_rotate(y, x, sps)

    # 6) 最终裁剪同长度
    final_sym = min(len(x), len(y)//sps)
    x = x[:final_sym]
    y = y[:final_sym*sps]

    # 7) 打包
    a = dict(
        samplerate = samplerate,
        symbolrate = baudrate,
        sps        = sps,
        distance   = distance,
        spans      = spans,
        cd         = cd,
        lpdbm      = lp_dbm,
        modformat  = modfmt,
        lag_sym    = int(lag_sym),
        phi_rad    = float(phi),
    )
    return Input(y=y, x=x, w0=0.0, a=a)

# ---------------- 公共 API ----------------
def load(mod, lp_dbm, rep, n_symbols=1_500_000):
    """Load single-channel 815 km PDM transmission data."""
    dat_grps, _ = labptptm1.select(mod, lp_dbm, rep)
    return [ _loader(dg, n_symbols, lp_dbm) for dg in
             tqdm(dat_grps, desc='loading LabPtPtm1', leave=False) ]
# ===================================================================

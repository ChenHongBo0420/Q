# === Singledata.py ==================================================
import zarr, numpy as np
from itertools import chain
from scipy.signal import correlate
from commplax import comm
from collections import namedtuple
from tqdm.auto import tqdm
import labptptm1

# ---------------- container ----------------
Input = namedtuple('DataInput', ['y', 'x', 'w0', 'a'])

# ---------------- public API ----------------
def load(mod, lp_dbm, rep, n_symbols=1_500_000):
    """Load single-channel 815 km PDM transmission data."""
    dat_grps, _ = labptptm1.select(mod, lp_dbm, rep)
    out = []
    for dg in tqdm(dat_grps, desc='loading LabPtPtm1', leave=False):
        out.append(_loader(dg, n_symbols, lp_dbm))
    return out


# ---------------- helper: metadata lookup ----------------
def _parent_group(grp):
    if not grp.path:                       # root
        return None
    p = '/'.join(grp.path.split('/')[:-1])
    return zarr.open_group(store=grp.store, path=p, mode='r')

def _get_meta(grp, *names):
    targets = {n.lower() for n in chain.from_iterable(
               n.split('|') for n in names)}
    g = grp
    while g is not None:
        for k, v in g.attrs.items():
            if k.lower() in targets:
                return v
        g = _parent_group(g)
    return None


# ---------------- helper: 对齐 & 相位补偿 ----------------
def _estimate_lag(y_ds, x_ref):
    """互相关估计整数符号 lag；recv 早 ⇒ lag<0。"""
    lag = (np.argmax(np.abs(correlate(y_ds, x_ref, mode='full')))
           - (len(x_ref) - 1))
    return lag

def _align_and_rotate(y, x, sps, max_sym=10_000):
    """裁剪 y/x 消除整数 lag，并校正平均相位。"""
    n_chk = min(max_sym, len(x))
    y_ds  = y[:n_chk*sps:sps]
    lag   = _estimate_lag(y_ds, x[:n_chk])

    if lag < 0:            # recv 早
        y = y[-lag*sps:]
        x = x[:len(y)//sps]
    elif lag > 0:          # recv 晚
        x = x[lag:]
        y = y[:len(x)*sps]

    # 常量相位误差
    phi = np.angle(np.mean(x[:n_chk] * np.conj(y[:n_chk*sps:sps])))
    y   = y * np.exp(1j*phi)
    return y, x, lag, phi


# ---------------- core loader ----------------
def _loader(dat_grp, n_symbols, lp_dbm):
    # 1) --- 元数据 ---
    baudrate   = _get_meta(dat_grp, 'baudrate|symbolrate')
    samplerate = _get_meta(dat_grp, 'samplerate|sample_rate|fs')
    if baudrate is None or samplerate is None:
        raise KeyError("Missing 'baudrate' or 'samplerate' in attrs tree.")
    sps = int(round(samplerate / baudrate))

    distance = _get_meta(dat_grp, 'distance')
    spans    = _get_meta(dat_grp, 'spans')
    modfmt   = _get_meta(dat_grp, 'modformat') or '16QAM'
    cd       = _get_meta(dat_grp, 'cd', 'dispersion') or 17e-6
    fo_hz    = _get_meta(dat_grp, 'fo', 'freq_offset') or 0.0   # 可能没有

    # 2) --- 读取波形 ---
    n_samples = int(n_symbols * sps)
    y = dat_grp['recv'][:n_samples]
    x = dat_grp['sent'][:n_symbols]

    # 3) --- CFO 粗补偿（如果有 fo） ---
    if fo_hz:
        t = np.arange(len(y))
        y = y * np.exp(-1j * 2*np.pi * fo_hz * t / samplerate)

    # 4) --- 去均值 + 功率归一化 + sent 缩放 ---
    y = y - np.mean(y, axis=0)
    y = comm.normpower(y, real=True) / np.sqrt(2)
    x = x / comm.qamscale(modfmt)

    # 5) --- 对齐 & 相位旋转 ---
    y, x, lag_sym, phi = _align_and_rotate(y, x, sps)

    # 6) --- 打包 metadata ---
    a = {
        'samplerate': samplerate,
        'symbolrate': baudrate,
        'sps':        sps,
        'distance':   distance,
        'spans':      spans,
        'cd':         cd,
        'lpdbm':      lp_dbm,
        'modformat':  modfmt,
        'lag_sym':    lag_sym,
        'phi_rad':    float(phi),
    }

    return Input(y=y, x=x, w0=0.0, a=a)
# =========================================================

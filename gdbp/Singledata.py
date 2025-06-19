# === Singledata.py ================================================
import zarr, numpy as np
from itertools import chain
from scipy.signal import correlate
from commplax import comm
from collections import namedtuple
from tqdm.auto import tqdm
import labptptm1

Input = namedtuple('DataInput', ['y', 'x', 'w0', 'a'])

# -------- metadata helpers --------
def _parent_group(grp):
    if not grp.path:
        return None
    par = '/'.join(grp.path.split('/')[:-1])
    return zarr.open_group(store=grp.store, path=par, mode='r')

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

# -------- align per polarization --------
def _estimate_lag(y, x):
    return (np.argmax(np.abs(correlate(y, x, 'full'))) - (len(x)-1))

def _align_and_rotate(y, x, sps, chk=12_000):
    """
    y: (Nsamp,2)  x: (Nsym,2)   -> 裁剪/旋转后对齐
    returns y,x,lag_list,phi
    """
    N = min(chk, len(x))
    lags = []
    # 1) 逐偏振 lag
    for p in (0,1):
        lag_p = _estimate_lag(y[:N*sps:sps, p], x[:N, p])
        lags.append(lag_p)
    lag0, lag1 = lags
    lag_min = min(lags)

    # 2) 裁剪两路
    offs0 = (lag0 - lag_min)*sps
    offs1 = (lag1 - lag_min)*sps
    y0 = y[offs0:, 0]
    y1 = y[offs1:, 1]
    L  = min(len(y0), len(y1))
    y  = np.stack([y0[:L], y1[:L]], axis=1)
    x  = x[-lag_min : -lag_min + L//sps]

    # 3) 平均相位
    phi = np.angle(np.mean(x[:N].flatten() *
                           np.conj(y[:N*sps:sps].flatten())))
    y *= np.exp(1j*phi)
    return y, x, lags, phi

# -------- core loader --------
def _loader(dg, n_sym, lp_dbm):
    br   = _get_meta(dg, 'baudrate|symbolrate')
    fs   = _get_meta(dg, 'samplerate|sample_rate|fs')
    if br is None or fs is None:
        raise KeyError("missing baudrate/samplerate")
    sps  = int(round(fs/br))

    mod  = _get_meta(dg, 'modformat') or '16QAM'
    beta = _get_meta(dg, 'rolloff') or 0.2

    # read
    y = dg['recv'][: (n_sym+4000)*sps ]
    x = dg['sent'][: n_sym]
    # normalise
    y = y - np.mean(y, axis=0)
    y = comm.normpower(y, real=True) / np.sqrt(2)
    x = x / comm.qamscale(mod)

    # align
    y, x, lags, phi = _align_and_rotate(y, x, sps)

    # pack
    a = dict(samplerate=fs, symbolrate=br, sps=sps,
             modformat=mod, rolloff=beta,
             lpdbm=lp_dbm, lag_list=lags, phi_rad=float(phi))
    return Input(y=y, x=x, w0=0.0, a=a)

def load(mod, lp_dbm, rep, n_symbols=1_500_000):
    grps, _ = labptptm1.select(mod, lp_dbm, rep)
    return [ _loader(g, n_symbols, lp_dbm) for g in
             tqdm(grps, desc='loading LabPtPtm1', leave=False) ]
# ===================================================================


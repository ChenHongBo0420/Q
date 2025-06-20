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

    # 3) 打包
    a = dict(
        samplerate = fs,
        symbolrate = br,
        sps        = sps,
        distance   = dist,
        spans      = spans,
        rolloff    = beta,
        modformat  = mod,
        lpdbm      = lp_dbm,
    )
    # 保持 w0=0（或你给的任意默认值）
    return Input(y=y, x=x, w0=0.0, a=a)


# -------------------------------------------------------------------
# public API
# -------------------------------------------------------------------
def load(mod, lp_dbm, rep, n_symbols=1_500_000):
    grps, _ = labptptm1.select(mod, lp_dbm, rep)
    return [ _loader(g, n_symbols, lp_dbm) for g in
             tqdm(grps, desc='loading LabPtPtm1', leave=False) ]
# ===================================================================

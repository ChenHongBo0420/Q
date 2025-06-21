# === gdbp/Singledata.py ============================================
import zarr
import numpy as np
from commplax import comm
from collections import namedtuple
from tqdm.auto import tqdm
import labptptm1
from itertools import chain

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
    # 1) extract basic metadata (walk up the tree if needed)
    br   = _get_meta(dg, 'baudrate|symbolrate')
    fs   = _get_meta(dg, 'samplerate|sample_rate|fs')
    dist = _get_meta(dg, 'distance', default=815e3)   # m
    spans= _get_meta(dg, 'spans|nspans', default=10)  # number of spans
    if br is None or fs is None:
        raise KeyError("missing 'baudrate' or 'samplerate'")

    # 2) compute samples-per-symbol
    sps = int(round(fs / br))

    # 3) read raw data
    y = dg['recv'][: n_sym * sps]
    x = dg['sent'][: n_sym]

    # 4) normalize power & remove DC
    y = y - np.mean(y, axis=0)
    y = comm.normpower(y, real=True) / np.sqrt(2)
    x = x / comm.qamscale(_get_meta(dg, 'modformat', default='16QAM'))

    # 5) build attribute dict, mirroring labptptm2 loader
    a = dict(
        samplerate   = fs,
        baudrate     = br,
        distance     = dist,
        spans        = spans,
        rolloff      = _get_meta(dg, 'rolloff', default=0.2),
        modformat    = _get_meta(dg, 'modformat', default='16QAM'),
        polmux       = _get_meta(dg, 'polmux', default=0),
        laserlinewidth = _get_meta(dg, 'laserlinewidth', default=100e3),
        chind        = _get_meta(dg, 'chind', default=0),
        srcid        = _get_meta(dg, 'srcid', default='src1'),
        lpdbm        = lp_dbm,
        lpw          = 10**(lp_dbm/10) / 1000.0,
        sps          = sps,
    )

    # 6) add total chromatic dispersion for 815 km link (ns/nm)
    #    CD_1125 = 18.451 ns/nm ⇒ D = 18.451/1125 ns/(nm·km) ≈ 0.016401
    #    CD_815  = D * 815 ≈ 0.016401 * 815 ≈ 13.367 ns/nm
    a['CD'] = 13.367

    # 7) default initial frequency offset (cycles/symbol)
    w0 = 0.0144

    return Input(y=y, x=x, w0=w0, a=a)

# -------------------------------------------------------------------
# public API
# -------------------------------------------------------------------
def load(mod, lp_dbm, rep, n_symbols=1500000):
    """
    Load the 815km dataset from labptptm1, producing a list of Inputs.
    
    Args:
        mod     : source sequence index (e.g. 1 or 2)
        lp_dbm  : launched power in dBm
        rep     : repeat index (1, 2, or 3)
        n_symbols: number of symbols to load (default 1_500_000)
    
    Returns:
        List of DataInput namedtuples with fields (y, x, w0, a).
    """
    grps, _ = labptptm1.select(mod, lp_dbm, rep)
    return [
        _loader(g, n_symbols, lp_dbm)
        for g in tqdm(grps, desc='loading 815km data', leave=False)
    ]
# ===================================================================

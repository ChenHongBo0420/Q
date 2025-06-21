# # === gdbp/Singledata.py ============================================
# import zarr
# import numpy as np
# from commplax import comm
# from collections import namedtuple
# from tqdm.auto import tqdm
# import labptptm1
# from itertools import chain

# Input = namedtuple('DataInput', ['y', 'x', 'w0', 'a'])

# # -------------------------------------------------------------------
# # metadata helpers
# # -------------------------------------------------------------------
# def _parent_group(grp):
#     if not grp.path:          # root
#         return None
#     par = '/'.join(grp.path.split('/')[:-1])
#     return zarr.open_group(store=grp.store, path=par, mode='r')

# def _get_meta(grp, *names, default=None):
#     targets = {n.lower() for n in chain.from_iterable(
#                n.split('|') for n in names)}
#     g = grp
#     while g is not None:
#         for k, v in g.attrs.items():
#             if k.lower() in targets:
#                 return v
#         g = _parent_group(g)
#     return default

# # -------------------------------------------------------------------
# # core loader
# # -------------------------------------------------------------------
# def _loader(dg, n_sym, lp_dbm):
#     # 1) extract basic metadata (walk up the tree if needed)
#     br   = _get_meta(dg, 'baudrate|symbolrate')
#     fs   = _get_meta(dg, 'samplerate|sample_rate|fs')
#     dist = _get_meta(dg, 'distance', default=815e3)   # m
#     spans= _get_meta(dg, 'spans|nspans', default=10)  # number of spans
#     if br is None or fs is None:
#         raise KeyError("missing 'baudrate' or 'samplerate'")

#     # 2) compute samples-per-symbol
#     sps = int(round(fs / br))

#     # 3) read raw data
#     y = dg['recv'][: n_sym * sps]
#     x = dg['sent'][: n_sym]

#     # 4) normalize power & remove DC
#     y = y - np.mean(y, axis=0)
#     y = comm.normpower(y, real=True) / np.sqrt(2)
#     x = x / comm.qamscale(_get_meta(dg, 'modformat', default='16QAM'))

#     # 5) build attribute dict, mirroring labptptm2 loader
#     a = dict(
#         samplerate   = fs,
#         baudrate     = br,
#         distance     = dist,
#         spans        = spans,
#         rolloff      = _get_meta(dg, 'rolloff', default=0.2),
#         modformat    = _get_meta(dg, 'modformat', default='16QAM'),
#         polmux       = _get_meta(dg, 'polmux', default=0),
#         laserlinewidth = _get_meta(dg, 'laserlinewidth', default=100e3),
#         chind        = _get_meta(dg, 'chind', default=0),
#         srcid        = _get_meta(dg, 'srcid', default='src1'),
#         lpdbm        = lp_dbm,
#         lpw          = 10**(lp_dbm/10) / 1000.0,
#         sps          = sps,
#     )

#     # 6) add total chromatic dispersion for 815 km link (ns/nm)
#     #    CD_1125 = 18.451 ns/nm ⇒ D = 18.451/1125 ns/(nm·km) ≈ 0.016401
#     #    CD_815  = D * 815 ≈ 0.016401 * 815 ≈ 13.367 ns/nm
#     a['CD'] = 13.367

#     # 7) default initial frequency offset (cycles/symbol)
#     w0 = 0.0144

#     return Input(y=y, x=x, w0=w0, a=a)

# # -------------------------------------------------------------------
# # public API
# # -------------------------------------------------------------------
# def load(mod, lp_dbm, rep, n_symbols=1500000):
#     """
#     Load the 815km dataset from labptptm1, producing a list of Inputs.
    
#     Args:
#         mod     : source sequence index (e.g. 1 or 2)
#         lp_dbm  : launched power in dBm
#         rep     : repeat index (1, 2, or 3)
#         n_symbols: number of symbols to load (default 1_500_000)
    
#     Returns:
#         List of DataInput namedtuples with fields (y, x, w0, a).
#     """
#     grps, _ = labptptm1.select(mod, lp_dbm, rep)
#     return [
#         _loader(g, n_symbols, lp_dbm)
#         for g in tqdm(grps, desc='loading 815km data', leave=False)
#     ]
# # ===================================================================

# === gdbp/Singledata.py ============================================
import zarr
import numpy as np
from commplax import comm
from collections import namedtuple
from tqdm.auto import tqdm
import labptptm1
from itertools import chain

# -------------------------------------------------------------------
# public data structure
# -------------------------------------------------------------------
Input = namedtuple('DataInput', ['y', 'x', 'w0', 'a'])

# -------------------------------------------------------------------
# metadata helpers
# -------------------------------------------------------------------
def _parent_group(grp):
    """Return parent zarr group; None if grp is root."""
    if not grp.path:  # root
        return None
    par = '/'.join(grp.path.split('/')[:-1])
    return zarr.open_group(store=grp.store, path=par, mode='r')

def _get_meta(grp, *names, default=None):
    """
    Walk up zarr hierarchy and query attributes by (case-insensitive) name(s).
    Supports `a|b|c` shorthand.
    """
    targets = {n.lower() for n in chain.from_iterable(n.split('|') for n in names)}
    g = grp
    while g is not None:
        for k, v in g.attrs.items():
            if k.lower() in targets:
                return v
        g = _parent_group(g)
    return default

# -------------------------------------------------------------------
# core loader for a single group
# -------------------------------------------------------------------
def _loader(dg, n_sym, lp_dbm):
    """Read raw recv/sent arrays, normalise, and pack into Input."""
    # 1) basic metadata ------------------------------------------------
    br    = _get_meta(dg, 'baudrate|symbolrate')
    fs    = _get_meta(dg, 'samplerate|sample_rate|fs')
    dist  = _get_meta(dg, 'distance',        default=815e3)   # [m]
    spans = _get_meta(dg, 'spans|nspans',    default=10)
    if br is None or fs is None:
        raise KeyError("missing 'baudrate' or 'samplerate' in metadata")

    # 2) samples-per-symbol -------------------------------------------
    sps = int(round(fs / br))

    # 3) read raw data -------------------------------------------------
    y = dg['recv'][: n_sym * sps]            # complex64, shape [Nsamples]
    x = dg['sent'][: n_sym]                  # complex64, shape [Nsymbols]

    # 4) normalisation -------------------------------------------------
    y = y - np.mean(y, axis=0)
    y = comm.normpower(y, real=True) / np.sqrt(2)
    x = x / comm.qamscale(_get_meta(dg, 'modformat', default='16QAM'))

    # 5) attribute dict (align with labptptm2 loader) ------------------
    a = dict(
        samplerate      = fs,
        baudrate        = br,
        distance        = dist,
        spans           = spans,
        rolloff         = _get_meta(dg, 'rolloff',        default=0.2),
        modformat       = _get_meta(dg, 'modformat',      default='16QAM'),
        polmux          = _get_meta(dg, 'polmux',         default=0),
        laserlinewidth  = _get_meta(dg, 'laserlinewidth', default=100e3),
        chind           = _get_meta(dg, 'chind',          default=0),
        srcid           = _get_meta(dg, 'srcid',          default='src1'),
        lpdbm           = lp_dbm,
        lpw             = 10 ** (lp_dbm / 10) / 1e3,
        sps             = sps,
        CD              = 13.367,             # for 815 km SSMF
    )

    # 6) default initial freq-offset (cycles/symbol) -------------------
    w0 = 0.0144

    return Input(y=y.astype(np.complex64),
                 x=x.astype(np.complex64),
                 w0=w0,
                 a=a)

# -------------------------------------------------------------------
# public API – single file, split into train / test
# -------------------------------------------------------------------
def load(mod: int,
         lp_dbm: float,
         rep: int,
         n_symbols: int = 1_500_000,
         ratio_train: float = 0.8,
         shuffle: bool = False,
         seed: int = 0):
    """
    Load one dataset from labptptm1 → split into train & test → return 2 Inputs.

    Parameters
    ----------
    mod        : source sequence index (1, 2, …)
    lp_dbm     : launched power [dBm]
    rep        : repeat index (1, 2, 3)
    n_symbols  : how many symbols to keep from the raw file
    ratio_train: fraction for training set (0 < ratio < 1)
    shuffle    : if True, randomly shuffle symbols before split
    seed       : RNG seed used when shuffle=True

    Returns
    -------
    [Input_train, Input_test]
    """
    if not (0.0 < ratio_train < 1.0):
        raise ValueError("ratio_train must be in (0,1)")

    # 0) query zarr group(s) -------------------------------------------
    grps, _ = labptptm1.select(mod, lp_dbm, rep)
    if len(grps) == 0:
        raise RuntimeError("labptptm1.select returned empty list")

    # 1) **只取第一组**；若想多组混合，改这里 --------------------------
    dg = grps[0]

    # 2) load full Input -----------------------------------------------
    full = _loader(dg, n_symbols, lp_dbm)

    # 3) split indices --------------------------------------------------
    N = full.x.shape[0]
    idx = np.arange(N)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)

    n_tr  = int(N * ratio_train)
    i_tr, i_te = idx[:n_tr], idx[n_tr:]

    # helper for slicing both y (sample domain) & x (symbol domain)
    def _slice(inp: Input, sel_sym):
        sel_sym = np.sort(sel_sym)           # ensure order
        sel_smp = np.concatenate([
            np.arange(s * inp.a['sps'], (s + 1) * inp.a['sps'])
            for s in sel_sym
        ])
        return Input(
            y  = inp.y[sel_smp],
            x  = inp.x[sel_sym],
            w0 = inp.w0,
            a  = inp.a.copy()
        )

    return [_slice(full, i_tr), _slice(full, i_te)]
# ====================================================================

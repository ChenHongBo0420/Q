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
Input = namedtuple('DataInput', ['y', 'x', 'bits', 'w0', 'a'])

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
# core loader for a single group  ★ 自动搜索 bits/src ★
# -------------------------------------------------------------------
def _loader(dg, n_sym, lp_dbm):
    """
    Read recv / sent / src;  auto-find src under any sub-folder like
    'source/16QAM65536/src' or '16qam65536/src'.
    """
    # -------- 1. metadata ----------
    br  = _get_meta(dg, 'baudrate|symbolrate')
    fs  = _get_meta(dg, 'samplerate|sample_rate|fs')
    if br is None or fs is None:
        raise KeyError("missing baudrate / samplerate in metadata")
    modfmt = _get_meta(dg, 'modformat', default='16QAM').upper()
    sps    = int(round(fs / br))

    # -------- 2. recv / sent -------
    y = dg['recv'][: n_sym * sps]
    x = dg['sent'][: n_sym]

    # -------- 3. find src ----------
    root = zarr.open_group(store=dg.store, path='/', mode='r')

    def _search_src(group, prefix=''):
        mfl = modfmt.lower()
        # 当前层 arrays
        for a in group.array_keys():
            path = f'{prefix}{a}'
            if 'src' in a.lower() and mfl in a.lower():
                return path
        # 递归子组
        for g in group.group_keys():
            sub = group[g]
            hit = _search_src(sub, f'{prefix}{g}/')
            if hit: return hit
        return None

    src_path = _search_src(root)
    if src_path is None:
        raise KeyError(f"No src bits found for {modfmt}")

    bits = zarr.open_array(root.store, src_path, mode='r')[: n_sym] \
               .astype(np.float32)                      # (n_sym, P)

    # -------- 4. normalise ---------
    y = comm.normpower(y - np.mean(y, axis=0), real=True) / np.sqrt(2)
    x = x / comm.qamscale(modfmt)

    # -------- 5. pack into Input ---
    a = dict(
        samplerate=fs, baudrate=br,
        sps=sps, modformat=modfmt,
        lpdbm=lp_dbm, lpw=10**(lp_dbm/10)/1e3,
        CD=13.367,  # …
    )
    w0 = 0.0144
    return Input(y=y.astype(np.complex64),
                 x=x.astype(np.complex64),
                 bits=bits,
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
        sel_sym = np.sort(sel_sym)
        sel_smp = np.concatenate([
            np.arange(s * inp.a['sps'], (s + 1) * inp.a['sps'])
            for s in sel_sym
        ])
        return Input(
            y    = inp.y[sel_smp],
            x    = inp.x[sel_sym],
            bits = inp.bits[sel_sym],          # ★ 同步裁剪 bits
            w0   = inp.w0,
            a    = inp.a.copy()
        )

    return [_slice(full, i_tr), _slice(full, i_te)]
# ====================================================================

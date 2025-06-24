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
    # ---------- 1. metadata ----------
    br  = _get_meta(dg, 'baudrate|symbolrate')
    fs  = _get_meta(dg, 'samplerate|sample_rate|fs')
    modfmt = (_get_meta(dg, 'modformat', default='16QAM') or '16QAM').upper()
    sps = int(round(fs / br))

    # ---------- 2. recv / sent -------
    y = dg['recv'][: n_sym * sps]
    x = dg['sent'][: n_sym]

    # ---------- 3. 自动定位 src bits ---
    root = zarr.open_group(store=dg.store, path='/', mode='r')
    mfl  = modfmt.lower()
    src_paths = []

    def walk_groups(grp, pref=''):
        # arrays在当前 group
        for a in grp.array_keys():
            if 'src' in a.lower():
                src_paths.append(('array', f'{pref}{a}'))
        # 子groups
        for g in grp.group_keys():
            gpath = f'{pref}{g}'
            if g.lower().endswith('src'):
                src_paths.append(('group', gpath))
            walk_groups(grp[g], gpath + '/')

    walk_groups(root)

    # 选择包含调制格式的路径；否则取第一个
    cand = [p for p in src_paths if mfl in p[1].lower()]
    kind_path = cand[0] if cand else (src_paths[0] if src_paths else None)

    if kind_path is None:
        print("⚠ src bits not found, fallback to sym_to_bits(sent)")
        from commplax.coding.FEC import sym_to_bits
        bits = sym_to_bits(x).astype(np.float32)
    else:
        kind, path = kind_path
        print(f"[Singledata] using src bits → {path}")
        if kind == 'array':
            bits = zarr.open_array(root.store, path)[: n_sym]
        else:
            g = zarr.open_group(root.store, path)
            subs = sorted(g.array_keys(), key=lambda s: int(s))
            bits = np.concatenate([g[k][:] for k in subs], axis=0)[: n_sym]
        bits = bits.astype(np.float32)

    # ---------- 4. normalise ----------
    y = comm.normpower(y - np.mean(y, axis=0), real=True) / np.sqrt(2)
    x = x / comm.qamscale(modfmt)

    # ---------- 5. pack ---------------
    a = dict(samplerate=fs, baudrate=br, sps=sps,
             modformat=modfmt, lpdbm=lp_dbm,
             lpw=10**(lp_dbm/10)/1e3)
    return Input(y.astype(np.complex64),
                 x.astype(np.complex64),
                 bits,
                 0.0144,   # w0
                 a)

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

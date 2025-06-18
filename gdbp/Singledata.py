import numpy as np
from commplax import comm
from collections import namedtuple
from tqdm.auto import tqdm
import labptptm1

# Container for the loaded data: received waveform, sent symbols, frequency offset, metadata
tuple_fields = ['y', 'x', 'w0', 'a']
Input = namedtuple('DataInput', tuple_fields)


def load(mod, lp_dbm, rep, n_symbols=1500000):
    """
    Load single-channel 815 km PDM transmission data.

    Args:
      mod:       modulation format index (0-based) or format-name string
      lp_dbm:    launched power in dBm
      rep:       repeat index (1,2,3,...)
      n_symbols: number of symbols to load

    Returns:
      list of Input tuples, each with fields (y, x, w0, a)
    """
    dat_grps, _ = labptptm1.select(mod, lp_dbm, rep)
    inputs = []
    for dg in tqdm(dat_grps, desc='loading LabPtPtm1 data', leave=False):
        inputs.append(_loader(dg, n_symbols, lp_dbm))
    return inputs

from itertools import chain

def _get_meta(grp, *names):
    """
    向上递归查找属性（大小写不敏感，支持别名）直到根组。若未找到返回 None
    在 Zarr 2.10 上对根组访问 .parent 会抛 ValueError，故捕获后终止。
    """
    names_lc = {n.lower() for n in chain.from_iterable(
        n.split('|') for n in names)}               # 允许写成 'baudrate|baud_rate'

    g = grp
    while g is not None:
        for k, v in g.attrs.items():
            if k.lower() in names_lc:
                return v
        # ---------- 向父组移动 ----------
        try:
            g = g.parent          # Zarr 2.10：根组 .parent -> ValueError
        except ValueError:
            g = None
    return None


def _loader(dat_grp, n_symbols, lp_dbm):
    # 1) 元数据（支持多别名，大小写无关）
    symbolrate = _get_meta(dat_grp, 'baudrate|baud_rate', 'symbolrate')
    samplerate = _get_meta(dat_grp, 'samplerate|sample_rate|fs')

    if symbolrate is None or samplerate is None:
        raise KeyError(
            "Cannot find 'baudrate'/'symbolrate' AND 'samplerate' in this group "
            "or any ancestor. Checked Zarr v2.10 hierarchy."
        )

    distance  = _get_meta(dat_grp, 'distance')
    spans     = _get_meta(dat_grp, 'spans')
    if distance is None or spans is None:
        raise KeyError("Missing 'distance' or 'spans' in attrs up the tree.")

    modformat = _get_meta(dat_grp, 'modformat') or '16QAM'
    cd        = _get_meta(dat_grp, 'cd', 'dispersion') or 17e-6

    # 2) 计算 samples-per-symbol
    sps = samplerate / symbolrate

    # 3) metadata dict
    a = {
        'symbolrate': symbolrate,
        'samplerate': samplerate,
        'sps':        sps,
        'distance':   distance,
        'spans':      spans,
        'cd':         cd,
        'lpdbm':      lp_dbm,
        'modformat':  modformat,
    }

    # 4) 读取波形
    n_samples = int(round(n_symbols * sps))
    y = dat_grp['recv'][:n_samples]
    x = dat_grp['sent'][:n_symbols]
    w0 = 0.0  # 单通道，未记录 FO

    # 5) 预处理
    y = y - np.mean(y, axis=0)
    y = comm.normpower(y, real=True) / np.sqrt(2)
    x = x / comm.qamscale(modformat)

    return Input(y=y, x=x, w0=w0, a=a)

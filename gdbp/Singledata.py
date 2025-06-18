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


def _get_meta(grp, *names):
    """
    向上递归地、大小写不敏感地查某些属性，直到根组。
    返回找到的第一个值或 None
    """
    names_lc = [n.lower() for n in names]
    g = grp
    while g is not None:
        for k, v in g.attrs.items():
            if k.lower() in names_lc:
                return v
        g = getattr(g, 'parent', None)   # Zarr >=2.18 有 parent
    return None


def _loader(dat_grp, n_symbols, lp_dbm):
    # ---------- 1) 获取属性（大小写无关 + 回溯） ----------
    symbolrate = _get_meta(dat_grp, 'baudrate', 'symbolrate', 'baud_rate')
    samplerate = _get_meta(dat_grp, 'samplerate', 'sample_rate', 'fs')

    if symbolrate is None or samplerate is None:
        raise KeyError(
            "Cannot find 'baudrate'/'symbolrate' and 'samplerate'. "
            "Checked current group and ancestors."
        )

    # ---------- 2) 其他元数据，亦可回溯 ----------
    distance  = _get_meta(dat_grp, 'distance')
    spans     = _get_meta(dat_grp, 'spans')
    modformat = _get_meta(dat_grp, 'modformat') or '16QAM'

    for k, v in {'distance': distance, 'spans': spans}.items():
        if v is None:
            raise KeyError(f"Missing '{k}' in attrs up the tree.")

    # ---------- 3) 计算 SPS ----------
    sps = samplerate / symbolrate

    # ---------- 4) 更新 metadata dict ----------
    a = {
        'symbolrate': symbolrate,
        'samplerate': samplerate,
        'sps':        sps,
        'distance':   distance,
        'spans':      spans,
        'cd':         _get_meta(dat_grp, 'cd', 'dispersion') or 17e-6,
        'lpdbm':      lp_dbm,
        'modformat':  modformat,
    }

    # ---------- 5) 读取波形 ----------
    n_samples = int(round(n_symbols * sps))
    y = dat_grp['recv'][:n_samples]
    x = dat_grp['sent'][:n_symbols]
    w0 = 0.0                      # 单通道没有额外 FO

    # ---------- 6) 预处理 ----------
    y = y - np.mean(y, axis=0)
    y = comm.normpower(y, real=True) / np.sqrt(2)
    x = x / comm.qamscale(modformat)


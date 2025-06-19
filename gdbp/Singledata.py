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

import zarr
from itertools import chain

def _parent_group(grp):
    """
    返回 grp 的父 Zarr Group（Zarr 2.10 无 .parent，用 path 推算）。
    根组返回 None。
    """
    if grp.path == '' or grp.path is None:   # 已到根
        return None
    parent_path = '/'.join(grp.path.split('/')[:-1])
    # open_group(path='') 会返回根组
    return zarr.open_group(store=grp.store, path=parent_path, mode='r')


def _get_meta(grp, *names):
    """
    向上递归查找属性（大小写无关，支持别名，用 '|' 分隔），
    兼容 Zarr 2.10（无 .parent）。
    """
    # 允许别名，如 'baudrate|baud_rate'
    names_lc = {n.lower() for n in chain.from_iterable(
        n.split('|') for n in names)}

    g = grp
    while g is not None:
        for k, v in g.attrs.items():
            if k.lower() in names_lc:
                return v
        g = _parent_group(g)   # 手动走到父组
    return None


def _loader(dat_grp, n_symbols, lp_dbm):
    # ---------- 1) 元数据 ----------
    symbolrate = _get_meta(dat_grp, 'baudrate|baud_rate', 'symbolrate')
    samplerate = _get_meta(dat_grp, 'samplerate|sample_rate|fs')
    if symbolrate is None or samplerate is None:
        raise KeyError("无法找到 'baudrate/symbolrate' 或 'samplerate'。")

    distance  = _get_meta(dat_grp, 'distance')
    spans     = _get_meta(dat_grp, 'spans')
    if distance is None or spans is None:
        raise KeyError("缺少 'distance' 或 'spans' 元数据。")

    modformat = _get_meta(dat_grp, 'modformat') or '16QAM'
    cd        = _get_meta(dat_grp, 'cd', 'dispersion') or 17e-6

    # ---------- 2) 计算 SPS ----------
    sps = samplerate / symbolrate

    # ---------- 3) metadata dict ----------
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

    # ---------- 4) 读取波形 ----------
    n_samples = int(round(n_symbols * sps))
    y = dat_grp['recv'][:n_samples]
    x = dat_grp['sent'][:n_symbols]
    w0 = 0.0  # 单通道

    # ---------- 5) 预处理 ----------
    y = y - np.mean(y, axis=0)
    y = comm.normpower(y, real=True) / np.sqrt(2)
    x = x / comm.qamscale(modformat)

    return Input(y=y, x=x, w0=w0, a=a)

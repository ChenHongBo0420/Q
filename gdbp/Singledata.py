import numpy as np
from collections import namedtuple
from tqdm.auto import tqdm
from commplax import comm
import labptptm1

# 数据容器：接收波形 y、发送符号 x、频偏 w0、元数据 a
Input = namedtuple('DataInput', ['y', 'x', 'w0', 'a'])

def load(mod, lp_dbm, rep, n_symbols=1500000):
    """
    Load single-channel 815 km PDM data.

    Args:
      mod:       modulation format index (0-based) or format-name string
      lp_dbm:    launched power in dBm (e.g. -6, -4, 0, 4)
      rep:       repetition index (1,2,3,…)
      n_symbols: number of symbols to load

    Returns:
      List[Input]: 每个 Input 包含 (y, x, w0, a)
    """
    dat_grps, _ = labptptm1.select(mod, lp_dbm, rep)
    inputs = []
    for dg in tqdm(dat_grps, desc='loading LabPtPtm1 data', leave=False):
        inputs.append(_loader(dg, n_symbols, lp_dbm))
    return inputs


def _loader(dat_grp, n_symbols, lp_dbm):
    """
    Internal loader: 依据 dat_grp.attrs 注入/补全元数据，读取 recv/sent，
    并做 DC 去除、归一化，返回 Input。
    """
    # 1) 拷贝组属性到普通 dict
    a = dict(dat_grp.attrs)

    # 2) 必要字段都从 attrs 里读
    symbolrate = a['baudrate']     # 28e9
    samplerate = a['samplerate']   # 56e9
    distance   = a['distance']     # 815e3
    spans      = a['spans']        # 10
    modformat  = a['modformat']    # e.g. '16QAM'

    # 3) 计算每符号采样数
    sps = samplerate / symbolrate

    # 4) 注入／补全 a 中的关键字段
    a.update({
        'symbolrate': symbolrate,
        'samplerate': samplerate,
        'sps':        sps,
        'distance':   distance,
        'spans':      spans,
        'cd':         a.get('cd', 17e-6),  # 如果没有，就设 17 ps/nm/km
        'lpdbm':      lp_dbm,              # 从参数传入
    })

    # 5) 根据 sps 把 recv 切到指定符号数
    n_samples = int(round(n_symbols * sps))
    y = dat_grp['recv'][:n_samples]
    x = dat_grp['sent'][:n_symbols]
    w0 = 0.0  # 单通道无频偏元数据

    # 6) 去 DC + 归一化
    y = y - np.mean(y, axis=0)
    y = comm.normpower(y, real=True) / np.sqrt(2)
    x = x / comm.qamscale(modformat)

    return Input(y=y, x=x, w0=w0, a=a)

import numpy as np
from commplax import comm
from . import store
from collections import namedtuple

Input = namedtuple('DataInput', ['y','x','w0','a'])

# 全局常量：若仓库里没给，就用下面这个 sps
DEFAULT_SPS       = 2.0
DEFAULT_SYMB_RATE = 28e9
DEFAULT_SAMP_RATE = DEFAULT_SPS * DEFAULT_SYMB_RATE

def _loader(dat_grp, n_symbols):
    # 1) 拿到属性字典
    a = dict(dat_grp.attrs)

    # 2) 尝试从 dat_grp.attrs 里取 sps
    if 'sps' in a:
        sps = a['sps']
    else:
        # 3) 尝试从 root.attrs 里取
        root = store.open_group()
        gattrs = dict(root.attrs)
        if 'sps' in gattrs:
            sps = gattrs['sps']
        elif 'symbolrate' in gattrs and 'samplerate' in gattrs:
            sps = gattrs['samplerate'] / gattrs['symbolrate']
        else:
            # 4) 回退到默认值
            sps = DEFAULT_SPS

    # 5) 读取数据
    n_samples = int(round(n_symbols * sps))
    y = dat_grp['recv'][:n_samples]
    x = dat_grp['sent'][:n_symbols]

    # 6) 因为单通道无 FO 信息，直接设 0
    w0 = 0.0

    # 7) 归一化
    y = y - np.mean(y, axis=0)
    y = comm.normpower(y, real=True) / np.sqrt(2)
    x = x / comm.qamscale(a.get('modformat', '16QAM'))

    return Input(y=y, x=x, w0=w0, a=a)

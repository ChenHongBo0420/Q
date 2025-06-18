import numpy as np
from collections import namedtuple
from tqdm.auto import tqdm
from commplax import comm
import zarr
import labptptm1

# Container for the loaded data: received, sent, freq offset, attributes
Input = namedtuple('DataInput', ['y', 'x', 'w0', 'a'])

def load(mod, lp_dbm, rep, n_symbols=1500000):
    """
    Load single-channel 815 km PDM transmission data.

    Args:
      mod:         modulation format index (0-based) or format name string
      lp_dbm:      launched power in dBm (e.g. -6, -4, -2, 0, 2, 4)
      rep:         repeat index (1, 2, or 3)
      n_symbols:   number of symbols to load (default 1.5e6)

    Returns:
      list of Input tuples, each with fields:
        y:   received waveform (samples)
        x:   transmitted symbols
        w0:  initial frequency offset (0.0 for single-channel)
        a:   metadata dict from Zarr group attrs
    """
    dat_grps, _ = labptptm1.select(mod, lp_dbm, rep)
    inputs = []
    for dg in tqdm(dat_grps, desc='loading LabPtPtm1 data', leave=False):
        inputs.append(_loader(dg, n_symbols))
    return inputs


def _loader(dat_grp, n_symbols):
    """
    Internal loader for a single Zarr Group.
    """
    # extract attributes
    a = dict(dat_grp.attrs)
    # samples per symbol
    sps = a['samplerate'] / a['baudrate']
    # determine number of samples to read
    n_samples = int(round(n_symbols * sps))
    # load arrays
    y = dat_grp['recv'][:n_samples]
    x = dat_grp['sent'][:n_symbols]
    # single-channel has no FO metadata
    w0 = 0.0
    # preprocessing: DC removal and normalization
    y = y - np.mean(y, axis=0)
    y = comm.normpower(y, real=True) / np.sqrt(2)
    x = x / comm.qamscale(a['modformat'])
    return Input(y=y, x=x, w0=w0, a=a)

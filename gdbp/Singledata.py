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


def _loader(dat_grp, n_symbols, lp_dbm):
    """
    Internal loader: reads recv/sent arrays, normalizes, and returns Input.
    """
    # 1) Extract attributes from Zarr group
    a = dict(dat_grp.attrs)

    # 2) Extract or fallback essential metadata
    symbolrate = a.get('baudrate', a.get('symbolrate', None))
    samplerate = a.get('samplerate', None)
    if symbolrate is None or samplerate is None:
        raise KeyError(f"Missing 'baudrate' or 'samplerate' in attrs: {list(a.keys())}")
    # 3) Compute samples-per-symbol
    sps = samplerate / symbolrate

    # 4) Link parameters
    distance  = a.get('distance', None)
    spans     = a.get('spans', None)
    if distance is None or spans is None:
        raise KeyError(f"Missing 'distance' or 'spans' in attrs: {list(a.keys())}")
    modformat = a.get('modformat', '16QAM')

    # 5) Update metadata dict with standard keys
    a.update({
        'symbolrate': symbolrate,
        'samplerate': samplerate,
        'sps':        sps,
        'distance':   distance,
        'spans':      spans,
        'cd':         a.get('cd', 17e-6),   # default CD if missing
        'lpdbm':      lp_dbm,
        'modformat':  modformat,
    })

    # 6) Load data arrays
    n_samples = int(round(n_symbols * sps))
    y = dat_grp['recv'][:n_samples]
    x = dat_grp['sent'][:n_symbols]
    w0 = 0.0  # no FO metadata for single-channel

    # 7) Preprocessing
    y = y - np.mean(y, axis=0)
    y = comm.normpower(y, real=True) / np.sqrt(2)
    x = x / comm.qamscale(modformat)

    return Input(y=y, x=x, w0=w0, a=a)

import numpy as np
from collections import namedtuple
from tqdm.auto import tqdm
from commplax import comm
import labptptm1

# Define a simple container for the loaded data
Input = namedtuple('DataInput', ['y', 'x', 'w0', 'a'])

def load(mod: Union[int, str], lp_dbm: int, rep: int, n_symbols: int = 1500000) -> List[Input]:
    """
    Load single-channel 815 km PDM transmission data.

    Args:
      mod:      modulation format index (0-based) or format name string
      lp_dbm:   launched power in dBm (e.g. -6, -4, -2, 0, 2, 4)
      rep:      repeat index (1, 2, or 3)
      n_symbols: number of symbols to load

    Returns:
      A list of Input tuples, each containing:
        y:   received waveform (samples)
        x:   transmitted symbols
        w0:  initial frequency offset (set to 0 for single-channel)
        a:   dict of metadata (attrs)
    """
    # Select the appropriate zarr groups
    dat_grps, _ = labptptm1.select(mod, lp_dbm, rep)

    inputs = []
    for dg in tqdm(dat_grps, desc='loading LabPtPtm1 data', leave=False):
        inputs.append(_loader(dg, n_symbols))
    return inputs


def _loader(dat_grp: zarr.Group, n_symbols: int) -> Input:
    """
    Internal loader for a single zarr Group.

    Extracts metadata, reads recv/sent arrays, and normalizes them.
    """
    # 1) Extract metadata
    a = dict(dat_grp.attrs)
    # 2) Calculate samples per symbol
    sps = a['samplerate'] / a['baudrate']
    # 3) Determine how many samples to fetch
    n_samples = int(round(n_symbols * sps))
    # 4) Load data arrays
    y = dat_grp['recv'][:n_samples]
    x = dat_grp['sent'][:n_symbols]
    # 5) For single-channel PDM we don't have FO metadata
    w0 = 0.0

    # 6) Preprocessing
    y = y - np.mean(y, axis=0)                           # DC removal
    y = comm.normpower(y, real=True) / np.sqrt(2)        # normalize power
    x = x / comm.qamscale(a['modformat'])                # QAM normalization

    return Input(y=y, x=x, w0=w0, a=a)

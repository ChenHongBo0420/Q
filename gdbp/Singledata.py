import numpy as np
from commplax import comm
from collections import namedtuple
from tqdm.auto import tqdm
import labptptm1

# DataInput holds received waveform, sent symbols, frequency offset, and metadata
tuple_fields = ['y','x','w0','a']
Input = namedtuple('DataInput', tuple_fields)

# Default samples-per-symbol if attrs do not specify
DEFAULT_SPS = 2.0


def load(mod, lp_dbm, rep, n_symbols=1500000):
    """
    Load single-channel 815 km PDM transmission data.

    Args:
      mod:       modulation format index (0-based) or format name string
      lp_dbm:    launched power in dBm
      rep:       repeat index (1,2,3)
      n_symbols: number of symbols to load

    Returns:
      list of Input tuples
    """
    # Select Zarr groups using labptptm1 API
    dat_grps, _ = labptptm1.select(mod, lp_dbm, rep)
    inputs = []
    for dg in tqdm(dat_grps, desc='loading LabPtPtm1 data', leave=False):
        inputs.append(_loader(dg, n_symbols))
    return inputs


def _loader(dat_grp, n_symbols):
    """
    Internal loader: reads recv/sent arrays, normalizes, and returns Input.
    """
    # Extract metadata attributes
    a = dict(dat_grp.attrs)
    # --- Inject required metadata for signal processing and DBP initialization ---
    # modulation format (default to 16QAM if missing)
    a.setdefault('modformat', '16QAM')
    # symbol rate (baudrate) and sample rate
    a.setdefault('symbolrate', 28e9)  # 28 GBd
    a.setdefault('samplerate', 56e9)  # 56 GSa/s
    # fiber properties
    a.setdefault('distance', 815e3)   # 815 km in meters
    a.setdefault('spans', 1)          # single span
    # dispersion parameter if needed (ps/nm/km converted to s/m^2)
    a.setdefault('cd', 17e-6)         # 17 ps/nm/km → 17e-6 s/m^2
    # launched power already in 'lpdbm'
    # ---------------------------------------------------------------
    # Determine samples-per-symbol
    sps = a.get('sps', DEFAULT_SPS)
    # Compute number of samples to read
    n_samples = int(round(n_symbols * sps))
    # Read arrays
    y = dat_grp['recv'][:n_samples]
    x = dat_grp['sent'][:n_symbols]
    # Single-channel has no frequency offset metadata
    w0 = 0.0
    # Preprocessing: DC removal and power normalization
    y = y - np.mean(y, axis=0)
    y = comm.normpower(y, real=True) / np.sqrt(2)
    # Normalize transmitted symbols
    x = x / comm.qamscale(a.get('modformat', '16QAM'))
    return Input(y=y, x=x, w0=w0, a=a)

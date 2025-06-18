# gdbp/__init__.py

# Expose the Singledata module and its load function
from . import Singledata
from .Singledata import load

# Expose auxiliary module
from . import aux

# Define public API
__all__ = [
    'Singledata',  # module
    'load',        # Singledata.load
    'aux',         # auxiliary utilities
]

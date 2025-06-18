# gdbp/__init__.py

# 先把 Singledata 模块本身暴露出来
from . import Singledata

# 如果你还想从包根直接拿到 load、help 等函数，也可以：
from .Singledata import load, help

# 如果还有 aux、commplax 之类，也都加进来：
from . import aux

# 定义 __all__，告诉 Python 这里有哪些可以被 import *
__all__ = [
    'Singledata',  # 模块本身
    'load',        # Singledata.load
    'help',        # Singledata.help
    'aux',         # 如果你要用 aux  
]

# -*- coding: utf-8 -*-
"""Vision.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1HzdeLpTQy6QR8kDo2_6FClJjoisQq_jP
"""

# Commented out IPython magic to ensure Python compatibility.
try:
  import jax
except ModuleNotFoundError:
#   %pip install --upgrade "jax[cpu]"
# install commplax if not found
try:
  import commplax
except ModuleNotFoundError:
#   %pip install https://github.com/ChenHongBo0420/Comm/archive/master.zip
# install data api if not found
try:
  import labptptm2
except ModuleNotFoundError:
#   %pip install https://github.com/remifan/LabPtPTm2/archive/master.zip


# install GDBP if not found
try:
  import gdbp
except ModuleNotFoundError:
#   %pip install https://github.com/ChenHongBo0420/Q/archive/main.zip

import jax
import numpy as np
import matplotlib.pyplot as plt
from commplax import equalizer as eq, xcomm, plot as cplt
from gdbp import gdbp_base as gb, data as gdat, plot as gplt
data = gdat.load(1, 0, 4, 2)[0]
data = gdat.load(1, 0, 4, 2)[0]
def train_gdbp(data, dtaps=261, ntaps=41):
  gdbp = gb.model_init(data, {'ntaps': ntaps, 'dtaps': dtaps, 'init_fn': gb.fdbp_init(data.a, xi=1.1, steps=3)}, [])
  return list(zip(*list(gb.train(gdbp, data, n_iter=2000))))


def train_fdbp(data, dtaps=261, ntaps=41):
  gdbp = gb.model_init(data, {'ntaps': ntaps, 'dtaps': dtaps, 'init_fn': gb.fdbp_init(data.a, xi=1.1, steps=3)}, [('fdbp_0',)])
  return list(zip(*list(gb.train(gdbp, data, n_iter=2000))))

def tree_diff(trees, **kwargs):
  diff = [jax.tree_map(lambda a, b: np.mean(np.abs(a - b)**2)/np.mean(np.abs(b)**2),
                       trees[i], trees[-1], **kwargs) for i in range(len(trees)-1)]
  return jax.tree_map(lambda *xs: list(xs), *diff)

loss_gdbp, params, state = train_gdbp(data)
params_diff = tree_diff(params)
state_diff = tree_diff(state)

import pprint

# 假设 params_diff 已经计算完成
pprint.pprint(params_diff)

import matplotlib.pyplot as plt

# 假设 params_diff 已经被正确计算并具有上述结构

# 提取参数差异数据
dconv_kernel = params_diff['parallel']['fdbp_series']['fdbp1']['DConv_0']['kernel']
rconv_kernel = params_diff['parallel']['fdbp_series']['RConv1']['kernel']
nconv_kernel = params_diff['parallel']['fdbp_series']['fdbp1']['NConv_0']['kernel']

# 创建绘图
plt.figure(figsize=(6, 4), dpi=300)
sli = slice(0, 2000)  # 根据实际数据长度调整

ax1 = plt.gca()
ax1.plot(dconv_kernel[sli], label=r'D-filter$_1$')
ax1.plot(rconv_kernel[sli], label=r'R-filter')
ax1.plot(nconv_kernel[sli], '--', label=r'N-filter$_1$', markersize=1)

# 添加图例和标签
ax1.legend(fontsize=8, loc='upper right')
ax1.set_ylabel(r'mean(|$\frac{\mathbf{\theta}_i - \mathbf{\theta}_{2000}}{\mathbf{\theta}_{2000}}|^2$)')
ax1.set_xlabel('Iteration $i$')

# 创建缩放插图
axins = ax1.inset_axes([0.4, 0.4, 0.3, 0.3])
axins.plot(dconv_kernel, label=r'D-filter$_1$')
axins.plot(rconv_kernel, label=r'R-filter')
axins.plot(nconv_kernel, '--', label=r'N-filter$_1$', markersize=1)

# 设置缩放图的范围
axins.set_xlim(0, 250)  # 根据需要调整
axins.set_ylim(0, 0.001)
axins.set_xticks([0, 250, 500])

# 标注缩放区域
ax1.indicate_inset_zoom(axins, edgecolor="slateblue")

# 显示绘图
plt.show()
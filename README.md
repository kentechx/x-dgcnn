<img src="./dgcnn.jpg" width="1200px"></img>

# x_dgcnn

A pytorch implementation of [DGCNN](https://arxiv.org/abs/1801.07829), more efficient and memory-saving than
[dgcnn.pytorch](https://github.com/antao97/dgcnn.pytorch).

Update: use LayerNorm and GELU other than BatchNorm and ReLU, except for the head which is using BatchNorm or
InstanceNorm for fast convergence. Rewrite the message passing part to make it more efficient. Move the normalization
and activation after the max pooling in all instances. Make the categorical embedding learnable.

[![PyPI version](https://badge.fury.io/py/x-dgcnn.svg)](https://badge.fury.io/py/x-dgcnn)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Installation

```bash
pip install x-dgcnn
```

## Usage

Classification.

```python
import torch
from x_dgcnn import DGCNN_Cls
model = DGCNN_Cls(k=20, in_dim=3, out_dim=10)
x = torch.randn(8, 2048, 3)
xyz = x.clone()
out = model(x, xyz)

```

Semantic segmentation.

```python
import torch
from x_dgcnn import DGCNN_Seg

model = DGCNN_Seg(k=40, in_dim=3, out_dim=10)
x = torch.randn(8, 2048, 3)
xyz = x.clone()
out = model(x, xyz)
```

Part segmentation.

```python
import torch
from x_dgcnn import DGCNN_Seg, SpatialTransformNet

stn = SpatialTransformNet(k=64)
model = DGCNN_Seg(k=40, in_dim=3, out_dim=10, n_category=10, stn=stn)
x = torch.randn(8, 2048, 3)
xyz = x.clone()
category = torch.randint(0, 10, (20,))
out = model(x, xyz, category)
```

## References

```bibtex
@article{wang2019dynamic,
  title={Dynamic graph cnn for learning on point clouds},
  author={Wang, Yue and Sun, Yongbin and Liu, Ziwei and Sarma, Sanjay E and Bronstein, Michael M and Solomon, Justin M},
  journal={Acm Transactions On Graphics (tog)},
  year={2019},
}
```

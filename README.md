# x_dgcnn

[![PyPI version](https://badge.fury.io/py/x-dgcnn.svg)](https://badge.fury.io/py/x-dgcnn)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Installation

```bash
pip install x-dgcnn
```

## Usage

Semantic segmentation.

```python
import torch
from x_dgcnn import DGCNN_Seg

model = DGCNN_Seg(k=64, in_dim=3, out_dim=10)
x = torch.randn(8, 2048, 3)
xyz = x
out = model(x, xyz)
```

Part segmentation.

```python
import torch
from x_dgcnn import DGCNN_Seg

model = DGCNN_Seg(k=64, in_dim=3, out_dim=10, n_category=10)
x = torch.randn(8, 2048, 3)
xyz = x
category = torch.randint(0, 10, (20,))
out = model(x, xyz, category)
```

## Citation

```bibtex
@article{wang2019dynamic,
  title={Dynamic graph cnn for learning on point clouds},
  author={Wang, Yue and Sun, Yongbin and Liu, Ziwei and Sarma, Sanjay E and Bronstein, Michael M and Solomon, Justin M},
  journal={Acm Transactions On Graphics (tog)},
  year={2019},
}
```

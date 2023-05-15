import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat, rearrange, einsum, reduce
from x_dgcnn.route import subset_topk


def exists(val):
    return val is not None


def pairwise_dists(x, y=None):
    # x: (b, n, d)
    # y: (b, m, d) or None
    if exists(y):
        xty = -2 * einsum(x, y, 'b n d, b m d -> b n m')
        xx = reduce(x ** 2, 'b n d -> b n 1', 'sum')
        yy = reduce(y ** 2, 'b m d -> b 1 m', 'sum')
        pair_dists = xx + xty + yy  # (b, n, m)
    else:
        inner = -2 * einsum(x, x, 'b n d, b m d -> b n m')
        xx = reduce(x ** 2, 'b n d -> b n 1', 'sum')
        pair_dists = xx + inner + xx.transpose(2, 1)  # (b, n, m)
    return pair_dists


def knn(x, k):
    # x: (b, n, d)
    neg_pdists = -1. * pairwise_dists(x)
    topk_ind = neg_pdists.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return topk_ind


class EdgeConv(nn.Module):
    def __init__(self, k, dims=(64, 64)):
        super().__init__()
        self.k = k

        dims = (dims[0] * 2, *dims)
        self.mlp = nn.Sequential(*[
            nn.Sequential(nn.Linear(in_d, out_d, bias=False),
                          nn.LayerNorm(out_d),
                          nn.GELU())
            for in_d, out_d in zip(dims[:-1], dims[1:])
        ])

    def route(self, x, neighbor_ind=None):
        # x: (b, n, d)
        d = x.shape[-1]
        if not exists(neighbor_ind):
            neighbor_ind = knn(x, k=self.k)  # (b, n, k)

        x = repeat(x, 'b n d -> b n k d', k=self.k)
        neighbor_ind = repeat(neighbor_ind, 'b n k -> b n k d', d=d)
        neighbor_x = x.gather(1, neighbor_ind)  # (b, n, k, d)

        # (b, n, k, d) -> (b, n, k, 2*d)
        graph_feature = torch.cat([neighbor_x - x, x], dim=-1)
        return graph_feature

    def forward(self, x, neighbor_ind=None):
        # x: (b, n, d)
        x = self.route(x, neighbor_ind)  # (b, n, k, 2*d)

        x = rearrange(x, 'b n k d -> b (n k) d')
        x = self.mlp(x)
        x = rearrange(x, 'b (n k) d -> b n k d', k=self.k)
        x = x.max(dim=2, keepdim=False)[0]  # (b, n, k, d) -> (b, n, d)
        return x


class SpatialTransformNet(nn.Module):
    """
    Spatial transformer network
    """

    def __init__(self, k, dims=(3, 64, 128)):
        super().__init__()
        self.k = k

        self.block = EdgeConv(k=k, dims=dims)

        self.mlp1 = nn.Sequential(
            nn.Linear(dims[-1], 1024, bias=False),
            nn.LayerNorm(1024),
            nn.GELU(),
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(1024, 256, bias=False),
            nn.LayerNorm(256),
            nn.GELU(),
        )

        self.transform = nn.Linear(256, 9)
        torch.nn.init.constant_(self.transform.weight, 0)
        torch.nn.init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x):
        # x: (b, n, d)
        x = self.block(x)  # (b, n, d) -> (b, n, d)
        x = self.mlp1(x)  # (b, n, d) -> (b, n, 1024)
        x = x.max(dim=1, keepdim=False)[0]  # (b, n, 1024) -> (b, 1024)
        x = self.mlp2(x)  # (b, 1024) -> (b, 1024)

        x = self.transform(x)  # (b, 256) -> (b, 9)
        x = x.reshape(-1, 3, 3)  # (b, 9) -> (b, 3, 3)
        return x


class DGCNN_Cls(nn.Module):
    def __init__(
            self,
            *,
            k,
            in_dim,
            out_dim,
            dims=(64, 64, 128, 256),
            emb_dim=1024,
            dynamic=True,
            dropout=0,
    ):
        super().__init__()
        self.k = k
        self.dynamic = dynamic

        dims = (in_dim, *dims)
        self.blocks = nn.ModuleList(
            [EdgeConv(k=k, dims=(di, do)) for di, do in zip(dims[:-1], dims[1:])]
        )

        self.mlp1 = nn.Sequential(
            nn.Linear(sum(dims[1:]), emb_dim, bias=False),
            nn.LayerNorm(emb_dim),
            nn.GELU(),
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.mlp2 = nn.Sequential(
            nn.Linear(emb_dim * 2, 512, bias=False),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(512, 256, bias=False),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(256, out_dim),
        )

    def forward(self, x, xyz):
        # x: (b, n, d)
        # xyz: (b, n, 3), spatial coordinates
        neighbor_ind = knn(xyz, k=self.k)  # (b, n, k)

        # go through all EdgeConv blocks
        xs = [self.blocks[0](x, neighbor_ind)]
        for block in self.blocks[1:]:
            x = block(xs[-1], None if self.dynamic else neighbor_ind)
            xs.append(x)
        x = torch.cat(xs, dim=-1)  # (b, n, sum(dims))

        # max & mean pooling
        x = self.mlp1(x)  # (b, n, sum(dims)) -> (b, n, emb_dim)
        x1 = x.max(dim=1, keepdim=False)[0]  # (b, n, emb_dim) -> (b, emb_dim)
        x2 = x.mean(dim=1, keepdim=False)  # (b, n, emb_dim) -> (b, emb_dim)
        x = torch.cat((x1, x2), dim=-1)  # (b, 2 * emb_dim)

        # mlp
        x = self.mlp2(self.dropout(x))  # (b, 2 * emb_dim) -> (b, out_dim)

        return x


class DGCNN_Seg(nn.Module):
    def __init__(
            self,
            *,
            k,
            in_dim,
            out_dim,
            emb_dim=1024,
            n_category=0,
            category_dim=64,
            depth=3,
            dynamic=True,
            dropout=0,
    ):
        super().__init__()
        self.k = k
        self.dynamic = dynamic

        # category embedding
        if n_category > 0:
            self.category_emb = nn.Embedding(n_category, category_dim)

        # EdgeConv blocks
        assert depth >= 2, 'depth must be >= 2'
        self.blocks = nn.ModuleList(
            [EdgeConv(k=k, dims=(in_dim, 64, 64))]
        )
        for _ in range(depth - 2):
            self.blocks.append(EdgeConv(k=k, dims=(64, 64, 64)))
        self.blocks.append(EdgeConv(k=k, dims=(64, 64)))

        # mlp1
        self.mlp1 = nn.Sequential(
            nn.Linear(depth * 64, emb_dim, bias=False),
            nn.LayerNorm(emb_dim),
            nn.GELU(),
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # mlp2
        dim = emb_dim + depth * 64 + (category_dim if n_category > 0 else 0)
        self.mlp2 = nn.Sequential(
            nn.Linear(dim, 256, bias=False),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(256, 128, bias=False),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x, xyz, category=None):
        # x: (b, n, d)
        # xyz: (b, n, 3), spatial coordinates
        n = x.size(1)
        neighbor_ind = knn(xyz, k=self.k)  # (b, n, k)

        xs = [self.blocks[0](x, neighbor_ind)]
        for block in self.blocks[1:]:
            x = block(xs[-1], None if self.dynamic else neighbor_ind)
            xs.append(x)
        x = torch.cat(xs, dim=-1)  # (b, n, depth * 64)

        # global feature
        x = self.mlp1(x)  # (b, n, d2) -> (b, n, emb_dim)
        x = x.max(1)[0]  # (b, n, emb_dim) -> (b, emb_dim)
        x = self.dropout(x)  # (b, emb_dim)
        if exists(category):
            x = torch.cat((x, self.category_emb(category)), dim=-1)

        # local feature
        x = repeat(x, 'b d -> b n d', n=n)  # (b, emb_dim) -> (b, n, emb_dim)
        x = torch.cat((x, *xs), dim=-1)  # (b, n, emb_dim + depth * 64)
        x = self.mlp2(x)  # (b, n, emb_dim + depth * 64) -> (b, n, out_dim)
        return x

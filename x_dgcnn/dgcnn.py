import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat, rearrange, einsum, reduce
from .route import subset_topk


def exists(val):
    return val is not None


class InstanceNorm1d(nn.Module):  # instance norm, done in the last dimension

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, 1, dim))
        self.b = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        var = x.var(dim=1, unbiased=False, keepdim=True)
        mean = x.mean(dim=1, keepdim=True)
        return self.g * (x - mean) / (var + self.eps) ** 0.5 + self.b


class EdgeConv(nn.Module):
    def __init__(self, k, dims=(64, 64)):
        super().__init__()
        self.k = k

        dims = (dims[0] * 2, *dims[1:])
        if len(dims) > 2:
            self.mlp = nn.Sequential(*[
                nn.Sequential(nn.Linear(in_d, out_d, bias=False),
                              nn.LayerNorm(out_d),
                              nn.GELU())
                for in_d, out_d in zip(dims[:-2], dims[1:-1])
            ])
            self.mlp.append(nn.Linear(dims[-2], dims[-1], bias=False))
        else:
            self.mlp = nn.Linear(dims[0], dims[1], bias=False)
        self.norm = nn.LayerNorm(dims[-1])
        self.act = nn.GELU()

    def route(self, x, neighbor_ind=None):
        # x: (b, n, d)
        d = x.shape[-1]
        if not exists(neighbor_ind):
            neighbor_ind = torch.cdist(x, x).topk(self.k, dim=-1, largest=False)[1]  # (b, n, k)

        x = repeat(x, 'b n d -> b n k d', k=self.k)
        neighbor_ind = repeat(neighbor_ind, 'b n k -> b n k d', d=d)
        neighbor_x = x.gather(1, neighbor_ind)  # (b, n, k, d)

        # (b, n, k, d) -> (b, n, k, 2*d)
        graph_feature = torch.cat([neighbor_x - x, x], dim=-1)
        return graph_feature

    def forward(self, x, neighbor_ind=None):
        # x: (b, n, d)
        x = self.route(x, neighbor_ind)  # (b, n, k, 2*d)

        x = self.mlp(x)
        x = x.max(dim=2, keepdim=False)[0]  # (b, n, k, d) -> (b, n, d)
        x = self.act(self.norm(x))
        return x


class SpatialTransformNet(nn.Module):
    """
    Spatial transformer network
    """

    def __init__(self, k, in_dim=3, head_bn=True):
        super().__init__()
        self.k = k

        dims = (in_dim, 64, 128)
        self.block = EdgeConv(k=k, dims=dims)

        norm = nn.BatchNorm1d if head_bn else nn.LayerNorm
        self.lin = nn.Linear(dims[-1], 1024, bias=False)

        self.norm = norm(1024)
        self.act = nn.GELU()

        self.head = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            norm(512),
            nn.GELU(),
            nn.Linear(512, 256, bias=False),
            norm(256),
            nn.GELU(),
            nn.Linear(256, 9)
        )

        torch.nn.init.constant_(self.head[-1].weight, 0)
        torch.nn.init.eye_(self.head[-1].bias.view(3, 3))

    def forward(self, x, neighbor_ind=None):
        # x: (b, n, d)
        x = self.block(x, neighbor_ind)  # (b, n, d) -> (b, n, d)

        x = self.lin(x)  # (b, n, d) -> (b, n, 1024)
        x = x.max(dim=1, keepdim=False)[0]  # (b, n, 1024) -> (b, 1024)
        x = self.act(self.norm(x))

        x = self.head(x)  # (b, 1024) -> (b, 9)
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
            head_bn=True,  # if using batchnorm in head, disable it if the batch size is 1
    ):
        super().__init__()
        self.k = k
        self.dynamic = dynamic

        dims = (in_dim, *dims)
        self.blocks = nn.ModuleList(
            [EdgeConv(k=k, dims=(di, do)) for di, do in zip(dims[:-1], dims[1:])]
        )

        self.lin = nn.Linear(sum(dims[1:]), emb_dim, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # BN is a linear operator on the feature space, whereas LN projects
        # the feature space onto a (d-2)-dimensional sphere which the mlp
        # head does not prefer.
        norm = nn.BatchNorm1d if head_bn else nn.LayerNorm
        self.norm = norm(emb_dim * 2)
        self.head = nn.Sequential(
            nn.Linear(emb_dim * 2, 512, bias=False),
            norm(512),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(512, 256, bias=False),
            norm(256),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(256, out_dim),
        )

    def forward(self, x, xyz):
        # x: (b, n, d)
        # xyz: (b, n, 3), spatial coordinates
        neighbor_ind = torch.cdist(xyz, xyz).topk(self.k, dim=-1, largest=False)[1]  # (b, n, k)

        # go through all EdgeConv blocks
        xs = [self.blocks[0](x, neighbor_ind)]
        for block in self.blocks[1:]:
            x = block(xs[-1], None if self.dynamic else neighbor_ind)
            xs.append(x)
        x = torch.cat(xs, dim=-1)  # (b, n, sum(dims))

        # max & mean pooling
        x = self.lin(x)  # (b, n, sum(dims)) -> (b, n, emb_dim)
        x1 = x.max(dim=1, keepdim=False)[0]  # (b, n, emb_dim) -> (b, emb_dim)
        x2 = x.mean(dim=1, keepdim=False)  # (b, n, emb_dim) -> (b, emb_dim)
        x = torch.cat((x1, x2), dim=-1)  # (b, 2 * emb_dim)

        # mlp
        x = self.head(self.dropout(self.norm(x)))  # (b, 2 * emb_dim) -> (b, out_dim)

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
            depth=3,
            stn: SpatialTransformNet = None,
            dynamic=True,
            dropout=0,
    ):
        super().__init__()
        self.k = k
        self.dynamic = dynamic

        # if using stn, put other features behind xyz
        self.stn = stn

        # EdgeConv blocks
        assert depth >= 2, 'depth must be >= 2'
        self.blocks = nn.ModuleList(
            [EdgeConv(k=k, dims=(in_dim, 64, 64))]
        )
        for _ in range(depth - 2):
            self.blocks.append(EdgeConv(k=k, dims=(64, 64, 64)))
        self.blocks.append(EdgeConv(k=k, dims=(64, 64)))

        # global linear
        self.lin = nn.Linear(depth * 64, emb_dim, bias=False)
        self.norm = nn.LayerNorm(emb_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # head
        dim = emb_dim + depth * 64
        self.mlp = nn.Sequential(
            nn.Linear(dim, 256, bias=False),
            InstanceNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(256, 256, bias=False),
            InstanceNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(256, 128, bias=False),
            InstanceNorm1d(128),
            nn.GELU(),
        )

        # category embedding
        if n_category > 0:
            self.category_emb = nn.Embedding(n_category, 128)
        self.head = nn.Linear(128, out_dim)

    def forward(self, x, xyz, category=None):
        # x: (b, n, d)
        # xyz: (b, n, 3), spatial coordinates
        n = x.size(1)
        neighbor_ind = torch.cdist(xyz, xyz).topk(self.k, dim=-1, largest=False)[1]  # (b, n, k)

        if exists(self.stn):
            transform = self.stn(x, neighbor_ind).reshape(-1, 3, 3)
            if x.size(2) > 3:
                x = torch.cat([torch.bmm(x[:, :, :3], transform), x[:, :, 3:]], dim=-1)
            else:
                assert x.size(2) == 3
                x = torch.bmm(x, transform)

        xs = [self.blocks[0](x, neighbor_ind)]
        for block in self.blocks[1:]:
            x = block(xs[-1], None if self.dynamic else neighbor_ind)
            xs.append(x)
        x = torch.cat(xs, dim=-1)  # (b, n, depth * 64)

        # global feature
        x = self.lin(x)  # (b, n, d2) -> (b, n, emb_dim)
        x = x.max(1)[0]  # (b, n, emb_dim) -> (b, emb_dim)
        x = self.dropout(self.act(self.norm(x)))  # (b, emb_dim)

        # local feature
        x = repeat(x, 'b d -> b n d', n=n)  # (b, emb_dim) -> (b, n, emb_dim)
        x = torch.cat((x, *xs), dim=-1)  # (b, n, emb_dim + depth * 64)
        x = self.mlp(x)  # (b, n, emb_dim + depth * 64) -> (b, n, 128)

        if exists(category):
            x = x + self.category_emb(category).unsqueeze(1)  # (b, n, 128) -> (b, n, 128)
        x = self.head(x)  # (b, n, 128) -> (b, n, out_dim)
        return x

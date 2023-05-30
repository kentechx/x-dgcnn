import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange, einsum, reduce
from collections import namedtuple
from dgl.geometry import farthest_point_sampler

from .dgcnn import cdist


def exists(val):
    return val is not None


def default(*vals):
    for val in vals:
        if exists(val):
            return val
    return None


def knn(x, k):
    # x: (b, n, m)
    selected_scores, selected_ind = cdist(x).topk(k, dim=-1, largest=False)  # (b, n, k)
    return selected_ind, selected_scores


def gather(x, ind):
    # x: (b, d, n)
    # ind: (b, m, k)
    # output: (b, d, m, k)
    m = ind.size(1)
    ind = repeat(ind, 'b m k -> b d (m k)', d=x.size(1))
    out = x.gather(-1, ind)  # (b, d, (m k))
    out = rearrange(out, 'b d (m k) -> b d m k', m=m)
    return out


class ChannelRMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim, 1))

    def forward(self, x):
        normed = F.normalize(x, dim=1)
        return normed * self.scale * self.gamma


class XEdgeConv(nn.Module):
    def __init__(self, k, dim=64, in_dim=None):
        super().__init__()
        self.k = k
        in_dim = default(in_dim, dim)

        self.norm1 = nn.BatchNorm1d(dim)
        self.lin1 = nn.Conv2d(2 * in_dim, dim, 1, bias=False)
        self.act1 = nn.GELU()

        self.lin2 = nn.Conv2d(2 * dim, dim, 1, bias=False)
        self.norm2 = nn.BatchNorm1d(dim)
        self.act2 = nn.GELU()

        self.shortcut = nn.Conv1d(in_dim, dim, 1, bias=False) if in_dim != dim else nn.Identity()

    def route(self, x, neighbor_ind):
        # x: (b, d, n)
        d = x.size(1)
        x = repeat(x, 'b d n -> b d n k', k=self.k)
        neighbor_ind = repeat(neighbor_ind, 'b n k -> b d n k', d=d)
        selected_x = x.gather(2, neighbor_ind)  # (b, d, n, k)

        # (b, d, n, k) -> (b, 2*d, n, k)
        graph_feature = torch.cat([selected_x - x, x], dim=1)
        return graph_feature

    def forward(self, x, neighbor_ind=None):
        # x: (b, d, n)
        if not exists(neighbor_ind):
            neighbor_ind, _ = knn(x, self.k)  # (b, n, k)
        input = x

        x = self.route(x, neighbor_ind)  # (b, 2*d, n, k)
        x = self.lin1(x).max(dim=-1, keepdim=False)[0]  # (b, 2*d, n, k) -> (b, d, n)
        x = self.act1(self.norm1(x))

        x = self.route(x, neighbor_ind)  # (b, 2*d, n, k)
        x = self.lin2(x).max(dim=-1, keepdim=False)[0]  # (b, 2*d, n, k) -> (b, d, n)

        return self.act2(self.norm2(self.shortcut(input) + x))


class CrossAttention(nn.Module):
    # modified from https://github.com/lucidrains/gigagan-pytorch
    def __init__(
            self,
            dim,
            dim_context,
            dim_head=64,
            heads=8
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads
        kv_input_dim = default(dim_context, dim)

        self.norm = ChannelRMSNorm(dim)
        self.norm_context = ChannelRMSNorm(kv_input_dim)

        self.to_q = nn.Conv1d(dim, dim_inner, 1, bias=False)
        self.to_kv = nn.Conv1d(kv_input_dim, dim_inner * 2, 1, bias=False)
        self.to_out = nn.Conv1d(dim_inner, dim, 1, bias=False)

    def forward(self, fmap, context, mask=None):
        """
        einstein notation

        b - batch
        h - heads
        n - fmap length
        m - context length
        d - dimension
        i - source seq (attend from)
        j - target seq (attend to)
        """
        # fmap: (b, d, n)
        # context: (b, d, m)

        fmap = self.norm(fmap)
        context = self.norm_context(context)

        h = self.heads

        q, k, v = (self.to_q(fmap), *self.to_kv(context).chunk(2, dim=1))

        k, v = map(lambda t: rearrange(t, 'b (h d) n -> (b h) n d', h=h), (k, v))

        q = rearrange(q, 'b (h d) n -> (b h) n d', h=self.heads)

        sim = -torch.cdist(q, k, p=2) * self.scale  # l2 distance

        if exists(mask):
            mask = repeat(mask, 'b j -> (b h) 1 j', h=self.heads)
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim=-1)

        out = einsum(attn, v, 'b i j, b j d -> b i d')

        out = rearrange(out, '(b h) n d -> b (h d) n', h=h)

        return self.to_out(out)


SampleResult = namedtuple('SampleResult', ['x', 'xyz', 'sample_ind', 'neighbor_ind'])


class FarthestPointSampler(nn.Module):

    def __init__(self, n_points, k=4):
        super().__init__()
        self.n_points = n_points
        self.k = k

    def forward(self, x, xyz, start_idx=None):
        # x: (b, d, n)
        # xyz: (b, 3, n)
        _xyz = rearrange(xyz, 'b d n -> b n d')
        sample_ind = farthest_point_sampler(_xyz, self.n_points, start_idx=start_idx)  # (b, k)
        sample_xyz = xyz.gather(-1, repeat(sample_ind, 'b k -> b d k', d=xyz.size(1)))  # (b, 3, k)

        neighbor_ind = cdist(sample_xyz, xyz).topk(self.k, dim=-1, largest=False)[1]  # (b, m, k)

        # recalculate sample_xyz: mean xyz
        sample_xyz = gather(xyz, neighbor_ind).mean(dim=-1, keepdim=False)  # (b, 3, m)
        sample_x = gather(x, neighbor_ind).max(dim=-1, keepdim=False)[0]  # (b, d, m)
        return SampleResult(sample_x, sample_xyz, sample_ind, neighbor_ind)


class XSampler(nn.Module):

    def __init__(self, n_points, dim, dim_context=None):
        super().__init__()
        dim_context = default(dim_context, dim)
        self.emb = nn.Parameter(torch.randn(1, dim, n_points))
        self.attn = CrossAttention(dim, dim_context)

    def forward(self, x):
        # x: (b, d, n)
        emb = repeat(self.emb, '1 d n -> b d n', b=x.size(0))
        x = self.attn(emb, x)  # (b, d, n_points)
        return SampleResult(x, x, None, None)


class XSpatialTransformNet(nn.Module):
    """
    Spatial transformer network
    """

    def __init__(self, k, in_dim=3, head_norm=True):  # disble head_norm if batchsize==1
        super().__init__()
        self.k = k

        self.mlp = nn.Sequential(
            nn.Conv1d(in_dim, 64, 1),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )

        self.block = XEdgeConv(k, dim=64)

        norm = nn.BatchNorm1d if head_norm else nn.Identity
        self.lin = nn.Conv1d(64, 512, 1, bias=False)

        self.norm = norm(512)
        self.act = nn.GELU()

        self.head = nn.Sequential(
            nn.Linear(512, 256, bias=False),
            norm(256),
            nn.GELU(),
            nn.Linear(256, 9)
        )

        torch.nn.init.constant_(self.head[-1].weight, 0)
        torch.nn.init.eye_(self.head[-1].bias.view(3, 3))

    def forward(self, x, neighbor_ind=None):
        # x: (b, d, n)
        x = self.mlp(x)  # (b, d, n) -> (b, 64, n)
        x = self.block(x, neighbor_ind)  # (b, d, n) -> (b, d, n)

        x = self.lin(x)  # (b, d, n) -> (b, 1024, n)
        x = x.max(dim=-1, keepdim=False)[0]  # (b, 1024, n) -> (b, 1024)
        x = self.act(self.norm(x))

        x = self.head(x)  # (b, 1024) -> (b, 9)
        return x


class XDGCNN_Cls(nn.Module):
    def __init__(
            self,
            *,
            in_dim,
            out_dim,
            sampler=FarthestPointSampler,
            base_points=4096,  # the number of input points, used to determine the number of hidden points
            sampling_ratio=(4, 16, 64, 256),  # the ratio of hidden points to input points
            k=8,
            dropout=0,
            head_norm=True,  # if using norm in head, disable it if the batch size is 1
    ):
        super().__init__()

        # projection
        self.mlp = nn.Sequential(
            nn.Conv1d(in_dim, 64, 1),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )

        self.first_k = 16
        self.conv = XEdgeConv(self.first_k, dim=64)

        # stages
        blocks = [1, 1, 1, 1]
        n_points = [base_points // r for r in sampling_ratio]
        dims = [64, 128, 256, 512]
        stages = []
        pre_dim = 64
        downsamples = []
        for i in range(len(blocks)):
            _k = min(n_points[i], k)
            if sampler is FarthestPointSampler:
                downsamples.append(FarthestPointSampler(n_points[i], _k))
            elif sampler is XSampler:
                downsamples.append(XSampler(n_points[i], pre_dim))
            else:
                raise NotImplementedError

            stages.append(nn.Sequential(
                *[XEdgeConv(_k, dim=dims[i], in_dim=pre_dim if _ii == 0 else dims[i])
                  for _ii in range(blocks[i])]))
            pre_dim = dims[i]

        self.downsamples = nn.ModuleList(downsamples)
        self.stages = nn.ModuleList(stages)

        # head
        norm = nn.BatchNorm1d if head_norm else nn.Identity
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(512, 256, bias=False),
            norm(256),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(256, out_dim),
        )

    def forward(self, x, xyz, start_idx=None):
        # x: (b, d, n)
        # xyz: (b, 3, n), spatial coordinates
        neighbor_ind = cdist(xyz).topk(self.first_k, dim=-1, largest=False)[1]  # (b, n, k)
        x = self.mlp(x)
        x = self.conv(x, neighbor_ind)

        # go through stages
        for downsample, stage in zip(self.downsamples, self.stages):
            if isinstance(downsample, FarthestPointSampler):
                sample_res = downsample(x, xyz, start_idx)
            elif isinstance(downsample, XSampler):
                sample_res = downsample(x)
            else:
                raise NotImplementedError
            x = stage(sample_res.x)
            xyz = sample_res.xyz

        x = x.mean(dim=-1, keepdim=False)  # (b, 512, n) -> (b, 512)
        x = self.head(self.dropout(x))  # (b, 512) -> (b, out_dim)

        return x

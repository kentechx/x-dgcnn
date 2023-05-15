import torch

from einops import repeat


def subset_topk(scores, k, tau=1., hard=False):
    """
    A differentiable subset topk operator.
    See https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/DL2/sampling/subsets.html
    :param scores: (..., n)
    :param k: The number of elements to select.
    :param tau: The temperature to control the softness of the topk operator.
    :param hard: If True, use straight through estimator to get a hard subset.
    :return:
    """
    _, top_k_ids = torch.topk(scores, k, dim=-1)
    _scores = repeat(scores, "... n -> ... n k", k=k)

    mask_value = -torch.finfo(_scores.dtype).max
    mask = torch.zeros_like(_scores, dtype=torch.bool)
    mask = mask.scatter_(-2, repeat(top_k_ids[..., :-1], '... m -> ... m k', k=k),
                         torch.ones_like(_scores, dtype=torch.bool).triu(1)[..., torch.arange(k - 1), :])
    _scores = _scores.masked_fill(mask, mask_value)
    khot = torch.softmax(_scores / tau, dim=1).sum(-1)

    if hard:
        # straight through
        khot_hard = torch.zeros_like(khot)
        val, ind = torch.topk(khot, k, dim=-1)
        khot_hard = khot_hard.scatter_(-1, ind, 1)
        res = khot_hard - khot.detach() + khot
    else:
        res = khot

    return res

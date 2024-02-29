import torch
from torch import Tensor

from elk.training.burns_norm import BurnsNorm
from ..utils.normparam import NormParam


def split_clusters(clusters: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
    """Split the clusters into negative and positive clusters."""
    x_neg = []
    x_pos = []
    for cluster_id, cluster in clusters.items():
        x_neg_, x_pos_ = cluster.unbind(1)

        x_neg.append(x_neg_)
        x_pos.append(x_pos_)

    return x_neg, x_pos


def cluster_norm(clusters: list[Tensor], return_params=False) -> Tensor | tuple[Tensor, list[NormParam]]:
    assert clusters[0].dim() == 2, "Expected shape (n, d)"
    xs = []
    if return_params:
        params = []
    for cluster in clusters:
        # Add a new dimension at index 1 for the template
        # this leads to the shape (n, 1, d)
        cluster = cluster.unsqueeze(1)

        norm = BurnsNorm(return_params=return_params)
        if return_params:
            hiddens_normalized, c_params = norm(cluster)
            params.append(c_params)
        else:
            hiddens_normalized = norm(cluster)

        xs.append(hiddens_normalized)

    if return_params:
        return torch.cat(xs, dim=0), params
    return torch.cat(xs, dim=0)

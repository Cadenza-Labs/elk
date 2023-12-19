import torch
from torch import Tensor
from tqdm import tqdm

from elk.training.burns_norm import BurnsNorm


def split_clusters(clusters: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
    """Split the clusters into negative and positive clusters."""
    x_neg = []
    x_pos = []
    for cluster_id, cluster in clusters.items():
        x_neg_, x_pos_ = cluster.unbind(1)

        x_neg.append(x_neg_)
        x_pos.append(x_pos_)

    return x_neg, x_pos


def cluster_norm(cluster):
    assert cluster[0].dim() == 2, "Expected shape (n, d)"
    xs = []
    for hiddens in tqdm(cluster):
        print("normalize by cluster")
        # Add a new dimension at index 1 for the template
        # this leads to the shape (n, 1, d)
        hiddens = hiddens.unsqueeze(1)

        norm = BurnsNorm()
        hiddens_normalized = norm(hiddens)

        xs.append(hiddens_normalized)

    return torch.cat(xs, dim=0)

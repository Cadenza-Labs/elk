import torch

from elk.training.burns_norm import BurnsNorm


def cluster_norm(clusters):
    assert clusters[0].dim() == 3, "Expected shape (n, k, d)"
    x_pos = []
    x_neg = []
    for cluster_id, hiddens in clusters.items():
        # Add a new dimension at index 1 for the template
        # this leads to the shape (n, 1, k,  d)
        hiddens = hiddens.unsqueeze(1)

        norm = BurnsNorm()
        hiddens_normalized = norm(hiddens)
        x_pos_, x_neg_ = hiddens_normalized.unbind(2)

        x_neg.append(x_neg_)
        x_pos.append(x_pos_)

    return torch.cat(x_neg, dim=0), torch.cat(x_pos, dim=0)

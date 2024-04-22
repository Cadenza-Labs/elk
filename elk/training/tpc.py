import torch
from sklearn.decomposition import PCA

from elk.normalization.cluster_norm import cluster_norm, split_clusters
from elk.training.burns_norm import BurnsNorm


def project_onto_top_pc(data):
    """
    Projects the data onto the top principal component.
    Args:
        data (numpy.array): The input data array. Shape: [n_samples, n_features]
    Returns:
        numpy.array: Data projected onto the top principal component.
    """
    pca = PCA(n_components=1)  # PCA for 1 principal component
    pca.fit(data)
    # top_pc = pca.components_[0]
    projections = pca.transform(data)
    return projections.flatten()


def pca_pytorch(data):
    """
    Perform PCA using PyTorch.
    """
    # Compute SVD
    U, S, V = torch.svd(data)

    # Extract the top principal component (first column of V)
    top_pc = V[:, 0]

    # Project data onto the top principal component
    projections = data @ top_pc

    return projections


def run_tpc(hiddens, labels, norm, device, layer):
    if norm is cluster_norm:
        true_x_neg, true_x_pos = split_clusters(hiddens)
        x_neg = cluster_norm(true_x_neg)
        x_pos = cluster_norm(true_x_pos)
        differences = (x_pos - x_neg).squeeze(1)
        labels = torch.cat([labels[key] for key in labels])
    elif type(norm) is BurnsNorm:
        differences = norm(hiddens[:, :, 0, :]) - norm(hiddens[:, :, 1, :])
        differences = differences.squeeze(1)  # remove the prompt template dimension
    else:
        raise NotImplementedError("Only cluster_norm and BurnsNorm are supported")

    assert differences.dim() == 2, "shape of differences has to be: (n, d)"

    projections = pca_pytorch(differences)
    crc_predictions = (projections > 0).to(device)

    estimate = labels.eq(crc_predictions).float().mean().item()
    estimate = max(estimate, 1 - estimate)
    print("layer", layer, "crc acc", estimate)

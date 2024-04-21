import torch
from sklearn.decomposition import PCA


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
    projections = torch.matmul(data, top_pc.unsqueeze(1)).squeeze()

    return projections

import torch
import torch.nn.functional as F


def compute_differences(contrast_features_pos, contrast_features_neg):
    """
    Compute the differences between pairs of positive and negative contrast features.

    Args:
    contrast_features_pos (torch.Tensor):
    Positive contrast features. Shape: [n_samples, n_features]
    contrast_features_neg (torch.Tensor):
    Negative contrast features. Shape: [n_samples, n_features]

    Returns:
    torch.Tensor: The differences between each pair of positive and negative features.
    """
    return contrast_features_pos - contrast_features_neg


def top_principal_component(data):
    """
    Finds the top principal component of the data.

    Args:
    data (torch.Tensor): The input data tensor. Shape: [n_samples, n_features]

    Returns:
    torch.Tensor: The top principal component. Shape: [n_features]
    """
    # Center the data
    mean = data.mean(0)
    data_centered = data - mean

    # Perform SVD
    U, S, V = torch.svd(data_centered)

    # Top principal component is the first right singular vector
    top_pc = V[:, 0]

    return top_pc


def project_onto_pc(data, pc):
    """
    Projects the data onto the specified principal component.

    Args:
    data (torch.Tensor): The input data tensor. Shape: [n_samples, n_features]
    pc (torch.Tensor): Principal component to project onto. Shape: [n_features]

    Returns:
    torch.Tensor: Data projected onto the principal component.
    """
    breakpoint()
    # Normalize the principal component to have unit norm
    pc_normalized = F.normalize(pc, p=2, dim=0)
    return torch.matmul(data, pc_normalized.unsqueeze(1)).squeeze()


# Example usage
if __name__ == "__main__":
    # Generate some dummy data for positive and negative contrast features
    n_samples = 100
    n_features = 20
    contrast_features_pos = torch.randn(n_samples, n_features)
    contrast_features_neg = torch.randn(n_samples, n_features)

    # Compute differences
    differences = compute_differences(contrast_features_pos, contrast_features_neg)

    # Find the top principal component of the differences
    top_pc = top_principal_component(differences)

    # Project the differences onto the top principal component
    projections = project_onto_pc(differences, top_pc)

    # Cluster based on the sign of the projection
    cluster_labels = torch.sign(projections)
    print(cluster_labels)

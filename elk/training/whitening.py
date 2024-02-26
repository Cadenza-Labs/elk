import torch
from torch import Tensor, nn

from warnings import warn

def whitening(x: Tensor) -> Tensor:
    """
    x : Tensor
        input of dimension (n, d)
    """
    n, d = x.shape
    mean = x.mean(dim=0)
    x = x - mean
    Sigma = torch.einsum("nd, nD -> dD", x, x) / (n - 1)
    U, Lambda, _ = torch.svd(Sigma)
    Lambda_sqrt_inv = torch.diag(1 / torch.sqrt(Lambda))
    W = U @ Lambda_sqrt_inv @ U.T
    #x_normalized = torch.einsum("nvD, vdD -> nvd", x_normalized, W)
    x = torch.einsum("nD, dD -> nd", x, W)

    return x


class WhitenNorm(nn.Module):
    """Whitening normalization."""

    def __init__(self, method: str = "zca", save_transform: bool = False):
        super().__init__()
        if method == "cholesky":
            warn("Cholesky whitening does not output a symmetric matrix. If you want to train a probe\
                   on the normalised space, you should not use this method as it is not mathematically guaranteed\
                   to have the same probe decisions in original and normalised space.")
        self.method = method
        self.W = None
        self.mean = None
        self.save_transform = save_transform

    def forward(self, x: Tensor) -> Tensor:
        """Normalizes per prompt template
        See table 1 of https://arxiv.org/pdf/1512.00809.pdf for more details
        Args:
            x: input of dimension (n, v, k, d) or (n, v, d)
        Returns:
            x_normalized: normalized output
        """
        # v is the number of clusters and we want to normalise each of them separately.
        # if x.dim() = 4, we want to normalise separately the positive and negative clusters (or on all the classes),
        # as done in the Burns paper whith only one cluster for the whole dataset.

        assert x.dim() in [3, 4], f"Input tensor must have dimension 3 or 4, but got {x.dim()}"
        
        if x.dim() == 3:
            n, v, d = x.shape
            x_normalized = x
        elif x.dim() == 4:
            n, v, k, d = x.shape
            x_normalized = x.view(n, v*k, d)
            v = k*v
        
        assert n >= d, f"Number of samples must be greater than the dimension of the data or the covariance cannot be inverted, but got {n} samples and {d} dimensions"

        if self.save_transform and self.W is not None and self.mean is not None:
            assert x_normalized.shape[1:] == self.mean.shape
            x_normalized = x_normalized - self.mean

            x_normalized = torch.einsum("nvD, vdD -> nvd", x_normalized, W)
            return x_normalized.view(x.shape)

        # TODO : what to do if n == 1 ? Or even n < d ? The covariance matrix is not invertible...
        mean = x_normalized.mean(dim=0)
        x_normalized: Tensor = x_normalized - mean

        Sigma = torch.einsum("nvd, nvD -> vdD", x_normalized, x_normalized) / (n - 1)

        if self.method in ["zca", "pca", "cholesky"]:
            # Sigma = U @ Lambda @ U^T
            U, Lambda, _ = torch.svd(Sigma)
            Lambda_sqrt_inv = torch.diag_embed(1 / torch.sqrt(Lambda))

            if self.method == "zca":
                # W = U @ Lambda^{-1/2} @ U.T
                W = torch.einsum("vde, vef -> vdf", U, Lambda_sqrt_inv)
                W = torch.einsum("vdf, vgf -> vdg", W, U) #transpose U

            elif self.method == "pca":
                # W = Lambda^{-1/2} @ U.T
                W = torch.einsum("vde, vfe -> vdf", Lambda_sqrt_inv, U) #transpose U

            elif self.method == "cholesky":
                # W = cholesky(U @ Lambda^{-1} @ U.T).T
                W = torch.einsum("vde, vef -> vdf", U, torch.diag_embed(1/Lambda))
                W = torch.einsum("vdf, vgf -> vdg", W, U) #transpose U
                W = torch.linalg.cholesky(W).transpose(-1, -2)

        elif self.method in ["zca_cor", "pca_cor"]:
            # Sigma = V^{1/2} @ P @ V^{1/2}
            V_sqrt_inv = torch.diag_embed(1 / torch.std(x_normalized, dim=0))
            P = torch.einsum("vdD, vDe -> vde", V_sqrt_inv, Sigma)
            P = torch.einsum("vde, vef -> vdf", P, V_sqrt_inv)

            # P = G @ Theta @ G.T
            G, Theta, _ = torch.svd(P)
            Theta_sqrt_inv = torch.diag_embed(1 / torch.sqrt(Theta))
            
            if self.method == "zca_cor":
                # W = P^{-1/2} @ V^{-1/2}
                W = torch.einsum("vde, vef -> vdf", G, Theta_sqrt_inv)
                W = torch.einsum("vdf, vgf -> vdg", W, G) #transpose G
                W = torch.einsum("vdg, vgh -> vdh", W, V_sqrt_inv)

            elif self.method == "pca_cor":
                # W = Theta^{-1/2} @ G.T @ V^{-1/2}
                W = torch.einsum("vde, vfe -> vdf", Theta_sqrt_inv, G) #transpose G
                W = torch.einsum("vdf, vfg -> vdg", W, V_sqrt_inv)

        x_normalized = torch.einsum("nvD, vdD -> nvd", x_normalized, W)

        if self.save_transform:
            self.W = W
            self.mean = mean

        return x_normalized.view(x.shape)
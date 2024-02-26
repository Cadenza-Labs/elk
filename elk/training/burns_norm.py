import torch
from torch import Tensor, nn


class BurnsNorm(nn.Module):
    """Burns et al. style normalization. Minimal changes from the original code."""

    def __init__(self, scale: bool = True, return_params: bool = False):
        super().__init__()
        self.scale: bool = scale
        self.return_params: bool = return_params

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
        """Normalizes per prompt template
        Args:
            x: input of dimension (n, v, k, d) or (n, v, d)
        Returns:
            x_normalized: normalized output
        """
        num_elements = x.shape[0]
        mean = x.mean(dim=0)
        x_normalized: Tensor = x - mean if num_elements > 1 else x

        if not self.scale:
            if self.return_params:
                return x_normalized, mean
            return x_normalized
        else:
            std = torch.linalg.norm(x_normalized, dim=0) / x_normalized.shape[0] ** 0.5
            assert std.dim() == x.dim() - 1

            # Compute the dimensions over which
            # we want to compute the mean standard deviation
            # exclude the first dimension v,
            # which is the template dimension
            dims = tuple(range(1, std.dim()))

            avg_norm = std.mean(dim=dims, keepdim=True)

            if self.return_params:
                return x_normalized / avg_norm, mean, avg_norm
            return x_normalized / avg_norm

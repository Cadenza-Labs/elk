from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import Tensor, nn, optim

from elk.normalization.cluster_norm import split_clusters


class PlattMixin(ABC):
    """Mixin for classifier-like objects that can be Platt scaled."""

    bias: nn.Parameter
    scale: nn.Parameter

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        ...

    def platt_scale(self, labels: Tensor, hiddens: Tensor, max_iter: int = 100):
        """Fit the scale and bias terms to data with LBFGS.

        Args:
            labels: Binary labels of shape [batch].
            hiddens: Hidden states of shape [batch, dim].
            max_iter: Maximum number of iterations for LBFGS.
        """
        opt = optim.LBFGS(
            [self.bias, self.scale],
            line_search_fn="strong_wolfe",
            max_iter=max_iter,
            tolerance_change=torch.finfo(hiddens.dtype).eps,
            tolerance_grad=torch.finfo(hiddens.dtype).eps,
        )

        def closure():
            opt.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(
                self(hiddens), labels.float()
            )

            loss.backward()
            return float(loss)

        opt.step(closure)

    def platt_scale_with_clusters(self, labels, clusters, max_iter: int = 100):
        opt = optim.LBFGS(
            [self.bias, self.scale],
            line_search_fn="strong_wolfe",
            max_iter=max_iter,
            tolerance_change=torch.finfo(clusters[0].dtype).eps,
            tolerance_grad=torch.finfo(clusters[0].dtype).eps,
        )

        x_neg, x_pos = split_clusters(clusters)

        hiddens = x_neg + x_pos

        labels = torch.cat(list(labels.values()), dim=0)
        labels = torch.cat(
            [labels, labels], dim=0
        )  # TODO: probably switch the labels for x_neg
        assert labels.dim() == 1, "Expected shape (n,)"

        # (_, v, k, d) = first_train_h.shape
        def closure():
            opt.zero_grad()
            # Normalize before that...
            reporter_output = self(hiddens)
            assert reporter_output.shape == labels.shape
            loss = nn.functional.binary_cross_entropy_with_logits(
                reporter_output, labels.float()
            )

            loss.backward()
            return float(loss)

        opt.step(closure)

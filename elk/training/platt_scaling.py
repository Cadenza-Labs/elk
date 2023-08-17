from abc import ABC, abstractmethod
from typing import Any

import torch
from einops import rearrange, repeat
from torch import Tensor, nn, optim

from elk.metrics import to_one_hot


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
        from elk.training import CcsReporter
        norm = None
        try:
            if self.config.platt_burns == "hack":
                norm = "hack"
        except Exception:
            print('not hack')

        n, v, k, d = hiddens.shape
        original_hiddens = hiddens
        original_labels = labels
        squashed_labels = to_one_hot(repeat(labels, "n -> (n v)", v=v), k).flatten()
        squashed_hiddens = rearrange(hiddens, "n v k d -> (n v k) d")

        opt = optim.LBFGS(
            [self.bias, self.scale],
            line_search_fn="strong_wolfe",
            max_iter=max_iter,
            tolerance_change=torch.finfo(squashed_hiddens.dtype).eps,
            tolerance_grad=torch.finfo(squashed_hiddens.dtype).eps,
        )

        def closure():
            if norm == "hack":
                opt.zero_grad()
                res = self(original_hiddens)
                print("res", res.shape)
                print('rearrange(res, "n v c -> (n v c)")', rearrange(res, "n v c -> (n v c)").shape)
                loss = nn.functional.binary_cross_entropy_with_logits(
                    rearrange(res, "n v c -> (n v c)"), squashed_labels.float()
                )
                loss.backward()
                return float(loss)
            else:
                opt.zero_grad()
                # reporter (***, d) -> (***)
                # (nvk, d) -> (nvk)
                print("squashed_hiddens", squashed_hiddens.shape)
                print("self(squashed_hiddens)", self(squashed_hiddens).shape)
                loss = nn.functional.binary_cross_entropy_with_logits(
                    self(squashed_hiddens), squashed_labels.float()
                )
                loss.backward()
                return float(loss)  

        opt.step(closure)


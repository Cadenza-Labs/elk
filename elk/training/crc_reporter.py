from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from einops import rearrange
from sklearn.metrics import accuracy_score

from elk.normalization.cluster_norm import cluster_norm, split_clusters
from elk.training.burns_norm import BurnsNorm
from elk.training.common import FitterConfig


@dataclass
class CrcConfig(FitterConfig):
    norm: Literal["burns", "cluster", "none"] = "none"
    num_layers: int = 1
    """The number of layers in the MLP."""

    cluster_algo: Literal["kmeans", "HDBSCAN", "spectral", None] = None
    k_clusters: int | None = None
    min_cluster_size: int | None = None


class CrcReporter(nn.Module):
    def __init__(
        self,
        config: CrcConfig,
        in_features: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.config = config
        self.linear = nn.Linear(in_features, 1, device=device, dtype=dtype)

    def fit(self, hiddens):
        # Normalize the hidden states
        if self.config.norm == "cluster":
            true_x_neg, true_x_pos = split_clusters(hiddens)
            x_neg = cluster_norm(true_x_neg)
            x_pos = cluster_norm(true_x_pos)
            differences = (x_pos - x_neg).squeeze(1)
        elif self.config.norm == "burns":
            norm = BurnsNorm()
            differences = norm(hiddens[:, :, 0, :]) - norm(hiddens[:, :, 1, :])
            # differences = differences.squeeze(1)
            # # remove the prompt template dimension
            differences = rearrange(
                differences, "n v d -> (n v) d", v=differences.shape[1]
            )  # remove the prompt template dimension
        else:
            raise NotImplementedError("Only cluster_norm and BurnsNorm are supported")

        assert differences.dim() == 2, "shape of differences has to be: (n, d)"

        _, _, vh = torch.pca_lowrank(differences, q=1, niter=10)

        # Use the TPC as the weight vector
        self.linear.weight.data = vh.T

    def forward(self, x):
        if self.config.norm == "cluster":
            x = cluster_norm(x)
        elif self.config.norm == "burns":
            norm = BurnsNorm()
            x = norm(x)
        else:
            print("self.config.norm", self.config.norm)
            print("No normalization")

        raw_scores = self.linear(x).squeeze()
        return raw_scores.squeeze(-1)

    def eval(self, hiddens, gt_labels):
        if self.config.norm == "cluster":
            gt_labels = torch.cat([gt_labels[key] for key in gt_labels])

            x_neg, x_pos = split_clusters(hiddens)
            predictions_pos = self(x_neg)
            predictions_neg = self(x_pos)
            predictions = torch.cat([predictions_neg, predictions_pos])
            gt_labels = gt_labels.repeat_interleave(2)

        elif self.config.norm == "burns":
            x_neg, x_pos = hiddens.unbind(dim=2)
            predictions = torch.cat([self(x_neg), self(x_pos)])
            v = predictions.shape[1]
            predictions = rearrange(predictions, "n v -> (n v)", v=v)
            gt_labels = gt_labels.repeat_interleave(2)
            gt_labels = gt_labels.repeat_interleave(v)

        predictions_binary = (predictions.detach().cpu().numpy() > 0).astype(bool)
        gt_labels_binary = gt_labels.detach().cpu().numpy().astype(bool)

        # Calculate accuracy using sklearn's accuracy_score
        acc = accuracy_score(gt_labels_binary, predictions_binary)
        estimate = max(acc, 1 - acc)
        print("crc acc", estimate)

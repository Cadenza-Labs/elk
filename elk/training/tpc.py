from dataclasses import dataclass
from typing import Literal

import torch

from elk.metrics.accuracy import AccuracyResult
from elk.metrics.eval import EvalResult
from elk.metrics.roc_auc import RocAucResult
from elk.normalization.cluster_norm import cluster_norm, split_clusters
from elk.training.burns_norm import BurnsNorm
from elk.training.common import FitterConfig


@dataclass
class CrcConfig(FitterConfig):
    norm: Literal["leace", "burns", "cluster", "none"] = "leace"
    cluster_algo: Literal["kmeans", "HDBSCAN", "spectral", None] = None
    k_clusters: int | None = None
    min_cluster_size: int | None = None


class CrcReporter(torch.nn.Module):
    config = CrcConfig

    def __init__(self, config: CrcConfig):
        super().__init__()
        self.config = config
        self.tpc = None

    def set_norm(self, norm):
        self.config.norm = norm

    def get_differences(self, hiddens):
        if self.config.norm == "cluster":
            true_x_neg, true_x_pos = split_clusters(hiddens)
            x_neg = cluster_norm(true_x_neg)
            x_pos = cluster_norm(true_x_pos)
            differences = (x_pos - x_neg).squeeze(1)
        elif self.config.norm == "burns":
            norm = BurnsNorm()
            differences = norm(hiddens[:, :, 0, :]) - norm(hiddens[:, :, 1, :])
            differences = differences.squeeze(1)  # remove the prompt template dimension
        elif self.config.norm == "none":
            differences = hiddens[:, :, 0, :] - hiddens[:, :, 1, :]
            # differences = rearrange(differences, "n v d -> (n v) d")

        assert differences.dim() == 2, "shape of differences has to be: (n, d)"
        return differences

    def fit(self, hiddens):
        differences = self.get_differences(hiddens)
        # Compute SVD
        # U, S, V = torch.svd(differences)
        U, S, V = torch.pca_lowrank(differences, niter=10, q=1)
        # Extract the top principal component (first column of V)
        self.tpc = V[:, 0]

    def forward(self, hiddens):
        differences = self.get_differences(hiddens)

        # Project the hiddens onto the top principal component
        projections = differences @ self.tpc
        crc_predictions = projections > 0
        # Predict the class label based on the sign of the projection
        return crc_predictions

    def eval(self, hiddens, gt_labels, layer):
        # if hiddens.dim() == 4:
        #     gt_labels = gt_labels.repeat_interleave(hiddens.shape[1])

        crc_predictions = self(hiddens)

        estimate = gt_labels.eq(crc_predictions).float().mean().item()
        estimate = max(estimate, 1 - estimate)
        # print("layer", layer, "crc acc", estimate)

        acc = AccuracyResult(
            estimate, 0, 0, 0
        )  # we just need the accuracy ... just put 0s for the rest
        auroc = RocAucResult(0, 0, 0)
        return EvalResult(acc, None, None, auroc, None)

import torch

from elk.training.burns_norm import BurnsNorm
from ..normalization.cluster_norm import cluster_norm, split_clusters

# TODO : Maybe you can use something like this to integrate into the CcsReporter for evaluation time
# Class to store global normalisation parameters to be used at evaluation time
class GlobalNorm:
    def __init__(self, mu_pos, mu_neg, sigma_pos, sigma_neg):
        self.mu_pos = mu_pos
        self.mu_neg = mu_neg
        self.sigma_pos = sigma_pos
        self.sigma_neg = sigma_neg

# TODO : I don't know how reporters work, but I think you can make this class into a reporter like CcsReporter
class CRC1:
    def __init__(self, in_features: int, method: str = "TPC"):
        self.in_features = in_features
        self.method = method
        self.direction = torch.zeros(in_features)
        self.global_norm = GlobalNorm(0, 0, 1, 1)
    
    # code to fit the CRC direction on the training data
    def fit(self, clusters: dict[str, torch.Tensor]):
        burnsNorm = BurnsNorm(return_params=True)

        x_neg, x_pos = split_clusters(clusters)
        stacked_x_neg = torch.cat([x for x in x_neg])
        stacked_x_pos = torch.cat([x for x in x_pos])
        _, mu_neg, sigma_neg = burnsNorm(stacked_x_neg)
        x_neg = cluster_norm(x_neg)
        _, mu_pos, sigma_pos = burnsNorm(stacked_x_pos)
        x_pos = cluster_norm(x_pos)

        self.global_norm = GlobalNorm(mu_pos, mu_neg, sigma_pos, sigma_neg)
        diff = (x_pos - x_neg).squeeze(1)
        if self.method == "TPC":
            self.direction = self._tpc(diff)
        elif self.method == "BSS":
            self.direction = self._bss(diff)
        else:
            raise ValueError(f"Unknown method {self.method}")
       
    def _tpc(self, diffs: torch.Tensor) -> torch.Tensor:
        """Compute the Top Principal Component of the differences."""
        U, S, V = torch.pca_lowrank(diffs, q=1, niter=10)
        self.temp_PCscores = U[:, 0] * S[0]
        return V[:, 0]
    
    def _bss(self, diffs: torch.Tensor):
        """Not implemented yet."""
        pass

    # code to evaluate the CRC on the test data
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Project the input onto the learned direction."""

        assert x.dim() == 3, "Expected shape (n, k, d)"

        x_neg, x_pos = x.unbind(1)
        x_neg = (x_neg - self.global_norm.mu_neg) / self.global_norm.sigma_neg
        x_pos = (x_pos - self.global_norm.mu_pos) / self.global_norm.sigma_pos

        diff = x_pos - x_neg

        return torch.einsum("nd,d->n", diff, self.direction)

    def test_train_acc(self, clusters: dict[str, torch.Tensor]) -> float:
        """Compute the accuracy of the CRC on the training data."""
        dict2list = [clusters[key] for key in clusters]
        x = torch.cat([x for x in dict2list])
        preds = self(x)
        preds = torch.sign(preds)
        gt = torch.sign(self.temp_PCscores)

        return (preds == gt).float().mean().item()

class CRC2:
    def __init__(self, in_features: int, method: str = "TPC"):
        self.in_features = in_features
        self.method = method
        self.direction = torch.zeros(in_features)
        self.global_norm = GlobalNorm(0, 0, 1, 1)
    
    # code to fit the CRC direction on the training data
    def fit(self, clusters: dict[str, torch.Tensor]):
        burnsNorm = BurnsNorm(return_params=True)

        x_neg, x_pos = split_clusters(clusters)
        stacked_x_neg = torch.cat([x for x in x_neg])
        stacked_x_pos = torch.cat([x for x in x_pos])
        _, mu_neg, sigma_neg = burnsNorm(stacked_x_neg)
        x_neg = cluster_norm(x_neg)
        _, mu_pos, sigma_pos = burnsNorm(stacked_x_pos)
        x_pos = cluster_norm(x_pos)

        self.global_norm = GlobalNorm(mu_pos, mu_neg, sigma_pos, sigma_neg)
        diff = (x_pos - x_neg).squeeze(1)
        if self.method == "TPC":
            self.direction = self._tpc(diff)
        elif self.method == "BSS":
            self.direction = self._bss(diff)
        else:
            raise ValueError(f"Unknown method {self.method}")
       
    def _tpc(self, diffs: torch.Tensor) -> torch.Tensor:
        """Compute the Top Principal Component of the differences."""
        U, S, V = torch.pca_lowrank(diffs, q=1, niter=10)
        self.temp_PCscores = U[:, 0] * S[0]
        return V[:, 0]
    
    def _bss(self, diffs: torch.Tensor):
        """Not implemented yet."""
        pass

    # code to evaluate the CRC on the test data
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Project the input onto the learned direction."""

        assert x.dim() == 3, "Expected shape (n, k, d)"

        x_neg, x_pos = x.unbind(1)
        #x_neg = (x_neg - self.global_norm.mu_neg) / self.global_norm.sigma_neg
        #x_pos = (x_pos - self.global_norm.mu_pos) / self.global_norm.sigma_pos

        diff = x_pos - x_neg

        return torch.einsum("nd,d->n", diff, self.direction)

    def test_train_acc(self, clusters: dict[str, torch.Tensor]) -> float:
        """Compute the accuracy of the CRC on the training data."""
        dict2list = [clusters[key] for key in clusters]
        x = torch.cat([x for x in dict2list])
        preds = self(x)
        preds = torch.sign(preds)
        gt = torch.sign(self.temp_PCscores)

        return (preds == gt).float().mean().item()

class CRC3:
    def __init__(self, in_features: int, method: str = "TPC"):
        self.in_features = in_features
        self.method = method
        self.direction = torch.zeros(in_features)
        self.global_norm = GlobalNorm(0, 0, 1, 1)
    
    # code to fit the CRC direction on the training data
    def fit(self, clusters: dict[str, torch.Tensor]):
        burnsNorm = BurnsNorm(return_params=True)

        x_neg, x_pos = split_clusters(clusters)
        stacked_x_neg = torch.cat([x for x in x_neg])
        stacked_x_pos = torch.cat([x for x in x_pos])
        x_neg, mu_neg, sigma_neg = burnsNorm(stacked_x_neg)
        #x_neg = cluster_norm(x_neg)
        x_pos, mu_pos, sigma_pos = burnsNorm(stacked_x_pos)
        #x_pos = cluster_norm(x_pos)

        self.global_norm = GlobalNorm(mu_pos, mu_neg, sigma_pos, sigma_neg)
        diff = (x_pos - x_neg).squeeze(1)
        if self.method == "TPC":
            self.direction = self._tpc(diff)
        elif self.method == "BSS":
            self.direction = self._bss(diff)
        else:
            raise ValueError(f"Unknown method {self.method}")
       
    def _tpc(self, diffs: torch.Tensor) -> torch.Tensor:
        """Compute the Top Principal Component of the differences."""
        U, S, V = torch.pca_lowrank(diffs, q=1, niter=10)
        self.temp_PCscores = U[:, 0] * S[0]
        return V[:, 0]
    
    def _bss(self, diffs: torch.Tensor):
        """Not implemented yet."""
        pass

    # code to evaluate the CRC on the test data
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Project the input onto the learned direction."""

        assert x.dim() == 3, "Expected shape (n, k, d)"

        x_neg, x_pos = x.unbind(1)
        x_neg = (x_neg - self.global_norm.mu_neg) / self.global_norm.sigma_neg
        x_pos = (x_pos - self.global_norm.mu_pos) / self.global_norm.sigma_pos

        diff = x_pos - x_neg

        return torch.einsum("nd,d->n", diff, self.direction)

    def test_train_acc(self, clusters: dict[str, torch.Tensor]) -> float:
        """Compute the accuracy of the CRC on the training data."""
        dict2list = [clusters[key] for key in clusters]
        x = torch.cat([x for x in dict2list])
        preds = self(x)
        preds = torch.sign(preds)
        gt = torch.sign(self.temp_PCscores)

        return (preds == gt).float().mean().item()

class CRC4:
    def __init__(self, in_features: int, method: str = "TPC"):
        self.in_features = in_features
        self.method = method
        self.direction = torch.zeros(in_features)
        self.global_norm = GlobalNorm(0, 0, 1, 1)
    
    # code to fit the CRC direction on the training data
    def fit(self, clusters: dict[str, torch.Tensor]):
        burnsNorm = BurnsNorm(return_params=True)

        x_neg, x_pos = split_clusters(clusters)
        stacked_x_neg = torch.cat([x for x in x_neg])
        stacked_x_pos = torch.cat([x for x in x_pos])
        x_neg, mu_neg, sigma_neg = burnsNorm(stacked_x_neg)
        #x_neg = cluster_norm(x_neg)
        x_pos, mu_pos, sigma_pos = burnsNorm(stacked_x_pos)
        #x_pos = cluster_norm(x_pos)

        self.global_norm = GlobalNorm(mu_pos, mu_neg, sigma_pos, sigma_neg)
        diff = (x_pos - x_neg).squeeze(1)
        if self.method == "TPC":
            self.direction = self._tpc(diff)
        elif self.method == "BSS":
            self.direction = self._bss(diff)
        else:
            raise ValueError(f"Unknown method {self.method}")
       
    def _tpc(self, diffs: torch.Tensor) -> torch.Tensor:
        """Compute the Top Principal Component of the differences."""
        U, S, V = torch.pca_lowrank(diffs, q=1, niter=10)
        self.temp_PCscores = U[:, 0] * S[0]
        return V[:, 0]
    
    def _bss(self, diffs: torch.Tensor):
        """Not implemented yet."""
        pass

    # code to evaluate the CRC on the test data
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Project the input onto the learned direction."""

        assert x.dim() == 3, "Expected shape (n, k, d)"

        x_neg, x_pos = x.unbind(1)
        #x_neg = (x_neg - self.global_norm.mu_neg) / self.global_norm.sigma_neg
        #x_pos = (x_pos - self.global_norm.mu_pos) / self.global_norm.sigma_pos

        diff = x_pos - x_neg

        return torch.einsum("nd,d->n", diff, self.direction)

    def test_train_acc(self, clusters: dict[str, torch.Tensor]) -> float:
        """Compute the accuracy of the CRC on the training data."""
        dict2list = [clusters[key] for key in clusters]
        x = torch.cat([x for x in dict2list])
        preds = self(x)
        preds = torch.sign(preds)
        gt = torch.sign(self.temp_PCscores)

        return (preds == gt).float().mean().item()
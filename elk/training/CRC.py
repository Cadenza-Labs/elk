import torch

from .burns_norm import BurnsNorm
from ..normalization.cluster_norm import cluster_norm, split_clusters
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pathlib

def create_pca_visualizations(hiddens, labels, special_direction, plot_name="pca_plot"):
    assert hiddens.dim() == 2, "reshape hiddens to (n, d)"

    # Use 3 components for PCA
    pca = PCA(n_components=3)
    reduced_data = pca.fit_transform(hiddens.cpu().numpy())

    # Create a 3D plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        reduced_data[:, 0],
        reduced_data[:, 1],
        reduced_data[:, 2],
        c=labels.cpu().numpy(),
        cmap="viridis",
    )

    pca_direction = pca.transform(special_direction.cpu().numpy())
    print(reduced_data[0])
    print(pca_direction)
    ax.quiver(0, 0, 0, pca_direction[0, 0], pca_direction[0, 1], pca_direction[0, 2], color="r")

    # Labeling the axes
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_zlabel("PCA Component 3")
    plt.title("PCA of Hidden Activations")

    # Saving the plot
    path = pathlib.Path(f"./pca_visualizations/{plot_name}.jpg")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close(fig)

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
    def __init__(self, in_features: int):
        self.in_features = in_features
        self.direction = torch.zeros(in_features)
        self.global_norm = GlobalNorm(0, 0, 1, 1)
    
    # code to fit the CRC direction on the training data
    def fit(self, clusters: dict[str, torch.Tensor], labels: dict[str, torch.Tensor]):
        burnsNorm = BurnsNorm(return_params=True)

        true_x_neg, true_x_pos = split_clusters(clusters)
        stacked_x_neg = torch.cat([x for x in true_x_neg])
        stacked_x_pos = torch.cat([x for x in true_x_pos])
        glob_neg, mu_neg, sigma_neg = burnsNorm(stacked_x_neg)
        x_neg = cluster_norm(true_x_neg)
        glob_pos, mu_pos, sigma_pos = burnsNorm(stacked_x_pos)
        x_pos = cluster_norm(true_x_pos)

        self.global_norm = GlobalNorm(mu_pos, mu_neg, sigma_pos, sigma_neg)
        diff = (x_pos - x_neg).squeeze(1)

        U, S, V = torch.pca_lowrank(diff, q=1, niter=10)
        self.temp_PCscores = U[:, 0] * S[0]
        self.direction = V[:, 0]
        
        # Create PCA visualizations
        gt_labels = torch.cat([labels[key] for key in labels])
        cluster_labels = torch.cat([torch.full_like(labels[key], int(key)) for key in labels])

        pca_labels = cluster_labels
        create_pca_visualizations(diff, pca_labels, self.direction.unsqueeze(0), "cluster_norm_diff")
        create_pca_visualizations(glob_pos - glob_neg, pca_labels, self.direction.unsqueeze(0), "burns_norm_diff")
        create_pca_visualizations(stacked_x_pos - stacked_x_neg, pca_labels, self.direction.unsqueeze(0), "true_diff")

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
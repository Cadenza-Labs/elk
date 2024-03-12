import pathlib

import plotly.express as px
import torch
from einops import rearrange
from sklearn.decomposition import PCA

from elk.training.burns_norm import BurnsNorm


def create_pca_visualizations(hiddens, labels, plot_name="pca_plot", out_dir="."):
    assert hiddens.dim() == 2, "reshape hiddens to (n, d)"

    # Use 3 components for PCA
    pca = PCA(n_components=3)
    reduced_data = pca.fit_transform(hiddens.cpu().numpy())

    # Create a 3D plot with Plotly Express
    fig = px.scatter_3d(
        x=reduced_data[:, 0],
        y=reduced_data[:, 1],
        z=reduced_data[:, 2],
        color=labels.cpu().numpy(),
        title="PCA of Hidden Activations",
        labels={"x": "PCA Component 1", "y": "PCA Component 2", "z": "PCA Component 3"},
    )

    # Saving the plot as an HTML file
    path = pathlib.Path(f"{out_dir}/pca_visualizations/{plot_name}.html")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path))


def pca_visualizations(layer, hiddens, train_gt, out_dir):
    n, v, k, d = hiddens.shape

    hiddens_difference = hiddens[:, :, 0, :] - hiddens[:, :, 1, :]
    hidden_mean = (hiddens[:, :, 0, :] + hiddens[:, :, 1, :]) / 2
    flattened_hiddens = rearrange(hiddens_difference, "n v d -> (n v) d", v=v)
    expanded_labels = train_gt.repeat_interleave(v)

    create_pca_visualizations(
        hiddens=flattened_hiddens,
        labels=expanded_labels,
        plot_name=f"diff_before_norm_{layer}",
        out_dir=out_dir,
    )

    create_pca_visualizations(
        hiddens=hidden_mean.view(-1, d),
        labels=expanded_labels,
        plot_name=f"mean_before_norm_{layer}",
        out_dir=out_dir,
    )

    # ... and after normalization
    norm = BurnsNorm()
    hiddens_neg, hiddens_pos = hiddens.unbind(2)
    normalized_hiddens_neg = norm(hiddens_neg)
    normalized_hiddens_pos = norm(hiddens_pos)
    normalized_hiddens = torch.stack(
        (normalized_hiddens_neg, normalized_hiddens_pos), dim=2
    )
    hiddens_difference = normalized_hiddens[:, :, 0, :] - normalized_hiddens[:, :, 1, :]
    hidden_mean = (normalized_hiddens[:, :, 0, :] + normalized_hiddens[:, :, 1, :]) / 2
    flattened_normalized_hiddens = rearrange(
        hiddens_difference, "n v d -> (n v) d", v=v
    )
    create_pca_visualizations(
        hiddens=flattened_normalized_hiddens,
        labels=expanded_labels,
        plot_name=f"diff_after_norm_{layer}",
        out_dir=out_dir,
    )

    create_pca_visualizations(
        hiddens=hidden_mean.view(-1, d),
        labels=expanded_labels,
        plot_name=f"mean_after_norm_{layer}",
        out_dir=out_dir,
    )

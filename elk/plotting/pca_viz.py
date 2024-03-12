import pathlib

import pandas as pd
import plotly.express as px
import torch
from sklearn.decomposition import PCA

from elk.normalization.cluster_norm import cluster_norm, split_clusters


def create_pca_visualizations(
    hiddens, color_labels, symbol_labels, plot_name="pca_plot", out_dir="."
):
    assert hiddens.dim() == 2, "reshape hiddens to (n, d)"

    # Use 3 components for PCA
    pca = PCA(n_components=3)
    reduced_data = pca.fit_transform(hiddens.cpu().numpy())

    # Combine color_labels and symbol_labels into a single label
    combined_labels = [f"{cl}_{sl}" for cl, sl in zip(color_labels, symbol_labels)]

    # Create a DataFrame for Plotly Express
    df = pd.DataFrame(
        {
            "PCA Component 1": reduced_data[:, 0],
            "PCA Component 2": reduced_data[:, 1],
            "PCA Component 3": reduced_data[:, 2],
            "Combined Label": combined_labels,
            "Symbol": symbol_labels.cpu().numpy(),
        }
    )

    # Create a custom color map
    color_map = {
        "0_0": "blue",  # dark color for cluster 0, gt label 0
        "0_1": "green",  # dark color for cluster 0, gt label 1
        "1_0": "red",  # light color for cluster 1, gt label 0
        "1_1": "black",  # light color for cluster 1, gt label 1
    }

    # Create a 3D plot with Plotly Express
    fig = px.scatter_3d(
        df,
        x="PCA Component 1",
        y="PCA Component 2",
        z="PCA Component 3",
        color="Combined Label",
        symbol="Symbol",
        color_discrete_map=color_map,
        title="PCA of Hidden Activations",
        labels={"x": "PCA Component 1", "y": "PCA Component 2", "z": "PCA Component 3"},
    )

    # Saving the plot as an HTML file
    path = pathlib.Path(f"{out_dir}/pca_visualizations/{plot_name}.html")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path))


def pca_visualizations_cluster(clusters, template_ids, labels, layer, out_dir):
    true_x_neg, true_x_pos = split_clusters(clusters)
    x_neg = cluster_norm(true_x_neg)
    x_pos = cluster_norm(true_x_pos)

    stacked_x_neg = torch.cat([x for x in true_x_neg])
    stacked_x_pos = torch.cat([x for x in true_x_pos])
    expanded_labels = torch.cat([labels[key] for key in labels])
    # expanded_labels = expanded_labels.cpu().numpy()
    # template_ids = torch.cat([torch.tensor(
    # template_ids[key]
    # ) for key in template_ids])

    diff = (x_pos - x_neg).squeeze(1)
    create_pca_visualizations(
        diff, template_ids, expanded_labels, f"cluster_norm_diff_{layer}", out_dir
    )
    create_pca_visualizations(
        stacked_x_pos - stacked_x_neg,
        template_ids,
        expanded_labels,
        f"true_diff_{layer}",
        out_dir,
    )


"""
def pca_visualizations(layer, hiddens, train_gt, out_dir):
    n, v, k, d = hiddens.shape

    hiddens_difference = hiddens[:, :, 0, :] - hiddens[:, :, 1, :]
    hidden_mean = (hiddens[:, :, 0, :] + hiddens[:, :, 1, :]) / 2
    flattened_hiddens = rearrange(hiddens_difference, "n v d -> (n v) d", v=v)
    expanded_labels = train_gt.repeat_interleave(v)

    create_pca_visualizations(
        hiddens=flattened_hiddens,
        color_labels=expanded_labels,
        plot_name=f"diff_before_norm_{layer}",
        out_dir=out_dir,
    )

    create_pca_visualizations(
        hiddens=hidden_mean.view(-1, d),
        color_labels=expanded_labels,
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
        color_labels=expanded_labels,
        plot_name=f"diff_after_norm_{layer}",
        out_dir=out_dir,
    )

    create_pca_visualizations(
        hiddens=hidden_mean.view(-1, d),
        color_labels=expanded_labels,
        plot_name=f"mean_after_norm_{layer}",
        out_dir=out_dir,
    )
    # code to fit the CRC direction on the training data
"""

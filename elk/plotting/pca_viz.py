import pathlib
import textwrap

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
from einops import rearrange
from sklearn.decomposition import PCA

from elk.normalization.cluster_norm import cluster_norm, split_clusters
from elk.training.burns_norm import BurnsNorm


def create_pca_visualizations(
    hiddens,
    labels,
    plot_name="pca_plot",
    out_dir=".",
    cluster_direction=None,
    burns_direction=None,
    force_direction=0,
    hover_labels=None,
):
    assert hiddens.dim() == 2, "reshape hiddens to (n, d)"

    # Use 3 components for PCA
    if force_direction > 0:
        # First, project the data onto the forced direction (1 for cluster, 2 for burns)
        if force_direction == 1:
            direction = cluster_direction
        else:
            direction = burns_direction
        scores = (hiddens @ direction).unsqueeze(1)
        projected_data = scores * direction.unsqueeze(0)
        orthogonal_data = hiddens - projected_data
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(orthogonal_data.cpu().numpy())
        reduced_data = (
            torch.cat([scores, torch.tensor(reduced_data, device=scores.device)], dim=1)
            .cpu()
            .numpy()
        )
    else:
        pca = PCA(n_components=3)
        reduced_data = pca.fit_transform(hiddens.cpu().numpy())

    # Create a 3D plot with Plotly Express
    dataframe = pd.DataFrame(reduced_data, columns=["PCA1", "PCA2", "PCA3"])
    dataframe["label"] = labels.cpu().numpy()

    def breaker(txt):
        return "<br>".join(textwrap.wrap(txt, width=90))

    if hover_labels is not None:
        hovertext = []
        for pair in hover_labels:
            hovertext.append(breaker(pair[0] + pair[1]))
    else:
        hovertext = None
    dataframe["hover_label"] = hovertext
    fig = px.scatter_3d(
        dataframe,
        x="PCA1",
        y="PCA2",
        z="PCA3",
        color="label",
        title="PCA of Hidden Activations",
        labels={"x": "PCA Component 1", "y": "PCA Component 2", "z": "PCA Component 3"},
        hover_data=["hover_label"],
    )

    if cluster_direction is not None:
        pca_cluster_dir = (
            pca.transform(cluster_direction.unsqueeze(0).cpu().numpy()) * 20
        )
        if force_direction > 0:
            dir_cluster_dir = (cluster_direction @ direction).unsqueeze(0).unsqueeze(
                1
            ).cpu() * 20
            pca_cluster_dir = (
                torch.cat(
                    [
                        dir_cluster_dir,
                        torch.tensor(pca_cluster_dir, device=dir_cluster_dir.device),
                    ],
                    dim=1,
                )
                .cpu()
                .numpy()
            )
        # Add an arrow for the cluster direction
        line = go.Scatter3d(
            x=[0, pca_cluster_dir[0, 0]],
            y=[0, pca_cluster_dir[0, 1]],
            z=[0, pca_cluster_dir[0, 2]],
            mode="lines+markers",
            line=dict(color="red", width=10),
            marker=dict(color="red", size=2),
            name="Cluster Direction",
        )
        cone = go.Cone(
            x=[pca_cluster_dir[0, 0]],
            y=[pca_cluster_dir[0, 1]],
            z=[pca_cluster_dir[0, 2]],
            u=[pca_cluster_dir[0, 0] * 0.5],
            v=[pca_cluster_dir[0, 1] * 0.5],
            w=[pca_cluster_dir[0, 2] * 0.5],
            # same color as line
            colorscale=[[0, "red"], [1, "red"]],
        )
        fig.add_trace(line)
        fig.add_trace(cone)

    if burns_direction is not None:
        pca_burns_dir = pca.transform(burns_direction.unsqueeze(0).cpu().numpy()) * 20
        if force_direction > 0:
            dir_burns_dir = (burns_direction @ direction).unsqueeze(0).unsqueeze(
                1
            ).cpu() * 20
            pca_burns_dir = (
                torch.cat(
                    [
                        dir_burns_dir,
                        torch.tensor(pca_burns_dir, device=dir_burns_dir.device),
                    ],
                    dim=1,
                )
                .cpu()
                .numpy()
            )
        # Add an arrow for the burns direction
        line = go.Scatter3d(
            x=[0, pca_burns_dir[0, 0]],
            y=[0, pca_burns_dir[0, 1]],
            z=[0, pca_burns_dir[0, 2]],
            mode="lines+markers",
            line=dict(color="green", width=10),
            marker=dict(color="green", size=2),
            name="Burns Direction",
        )
        cone = go.Cone(
            x=[pca_burns_dir[0, 0]],
            y=[pca_burns_dir[0, 1]],
            z=[pca_burns_dir[0, 2]],
            u=[pca_burns_dir[0, 0] * 0.5],
            v=[pca_burns_dir[0, 1] * 0.5],
            w=[pca_burns_dir[0, 2] * 0.5],
            # same color as line
            colorscale=[[0, "green"], [1, "green"]],
        )
        fig.add_trace(line)
        fig.add_trace(cone)

        torch.cosine_similarity(
            torch.tensor(pca_cluster_dir[0]), torch.tensor(pca_burns_dir[0]), dim=0
        ).item()

    # Saving the plot as an HTML file
    path = pathlib.Path(f"{out_dir}/pca_visualizations/{plot_name}.html")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path))


def pca_visualizations_cluster(layer, clusters, gt_labels, out_dir, hover_labels=None):
    true_x_neg, true_x_pos = split_clusters(clusters)
    x_neg = cluster_norm(true_x_neg)
    x_pos = cluster_norm(true_x_pos)
    cluster_diff = (x_pos - x_neg).squeeze(1)

    U, S, V = torch.pca_lowrank(cluster_diff, q=1, niter=10)
    cluster_direction = V[:, 0]
    cluster_logits = U[:, 0] * S[0]
    cluster_preds = torch.sign(cluster_logits)

    stacked_x_neg = torch.cat([x for x in true_x_neg])
    stacked_x_pos = torch.cat([x for x in true_x_pos])
    stacked_labels = torch.cat([gt_labels[key] for key in gt_labels])
    true_diff = stacked_x_pos - stacked_x_neg
    true_diff -= true_diff.mean(dim=0)

    burns_pos = BurnsNorm()(stacked_x_pos)
    burns_neg = BurnsNorm()(stacked_x_neg)
    burns_diff = burns_pos - burns_neg
    burns_diff -= burns_diff.mean(dim=0)

    U, S, V = torch.pca_lowrank(burns_diff, q=1, niter=10)
    burns_direction = V[:, 0]
    burns_logits = U[:, 0] * S[0]
    burns_preds = torch.sign(burns_logits)

    torch.cat([torch.full_like(gt_labels[key], int(key)) for key in gt_labels])
    # pca_labels = cluster_labels
    pca_labels = stacked_labels

    create_pca_visualizations(
        cluster_diff,
        pca_labels,
        f"cluster_norm_diff_{layer}",
        out_dir,
        cluster_direction,
        burns_direction,
        force_direction=1,
        hover_labels=hover_labels,
    )
    create_pca_visualizations(
        burns_diff,
        pca_labels,
        f"burns_norm_diff_{layer}",
        out_dir,
        cluster_direction,
        burns_direction,
        force_direction=2,
        hover_labels=hover_labels,
    )
    create_pca_visualizations(
        true_diff,
        pca_labels,
        f"true_diff_centered_{layer}",
        out_dir,
        cluster_direction,
        burns_direction,
        force_direction=0,
        hover_labels=hover_labels,
    )

    # Compute the prediction of both directions on their respectively
    # normalized data and compare it to the other normalized or not data
    agreement = torch.sum(cluster_preds == burns_preds).item() / len(cluster_preds)

    ccpred = cluster_diff @ cluster_direction
    cbpred = burns_diff @ cluster_direction
    cnpred = true_diff @ cluster_direction
    bcpred = cluster_diff @ burns_direction
    bbpred = burns_diff @ burns_direction
    bnpred = true_diff @ burns_direction

    ccn = torch.sum(cluster_preds == torch.sign(ccpred)).item() / len(cluster_preds)
    cbn = torch.sum(cluster_preds == torch.sign(cbpred)).item() / len(cluster_preds)
    cnn = torch.sum(cluster_preds == torch.sign(cnpred)).item() / len(cluster_preds)
    bcn = torch.sum(burns_preds == torch.sign(bcpred)).item() / len(burns_preds)
    bbn = torch.sum(burns_preds == torch.sign(bbpred)).item() / len(burns_preds)
    bnn = torch.sum(burns_preds == torch.sign(bnpred)).item() / len(burns_preds)

    cca = torch.mean((stacked_labels == (ccpred > 0).float()).float())
    cba = torch.mean((stacked_labels == (cbpred > 0).float()).float())
    c_n_a = torch.mean((stacked_labels == (cnpred > 0).float()).float())
    bca = torch.mean((stacked_labels == (bcpred > 0).float()).float())
    bba = torch.mean((stacked_labels == (bbpred > 0).float()).float())
    bna = torch.mean((stacked_labels == (bnpred > 0).float()).float())

    # save the agreement table
    with open(f"{out_dir}/pca_visualizations/{layer}_agreement.txt", "w") as f:
        f.write(
            f"CRC directions cosine similarity: "
            f"{torch.cosine_similarity(cluster_direction, burns_direction, dim=0).item()}\n"  # noqa
        )
        f.write(f"CRC agreement: {agreement}\n")
        f.write("Cluster direction consistency:\n")
        f.write("Prediction change :")
        f.write(f"\tCluster norm: {ccn}\n")
        f.write(f"\tBurns norm: {cbn}\n")
        f.write(f"\tNo norm: {cnn}\n")
        f.write("Accuracy :\n")
        f.write(f"\tCluster norm: {cca}\n")
        f.write(f"\tBurns norm: {cba}\n")
        f.write(f"\tNo norm: {c_n_a}\n")  # noqa
        f.write("Burns direction consistency:\n")
        f.write("Prediction change :")
        f.write(f"\tCluster norm: {bcn}\n")
        f.write(f"\tBurns norm: {bbn}\n")
        f.write(f"\tNo norm: {bnn}\n")
        f.write("Accuracy :\n")
        f.write(f"\tCluster norm: {bca}\n")
        f.write(f"\tBurns norm: {bba}\n")
        f.write(f"\tNo norm: {bna}\n")


def pca_visualizations(layer, hiddens, train_gt, out_dir, hover_labels=None):
    if hiddens.dim() == 3:
        hiddens = hiddens.unsqueeze(1)
    n, v, k, d = hiddens.shape

    hiddens_difference = hiddens[:, :, 0, :] - hiddens[:, :, 1, :]
    hidden_mean = (hiddens[:, :, 0, :] + hiddens[:, :, 1, :]) / 2
    flattened_hiddens = rearrange(hiddens_difference, "n v d -> (n v) d", v=v)
    expanded_labels = train_gt.repeat_interleave(v)
    duplicated_labels = torch.stack(
        [train_gt.view(-1, 1, 1), train_gt.view(-1, 1, 1)], dim=2
    )

    create_pca_visualizations(
        hiddens=flattened_hiddens,
        labels=expanded_labels,
        plot_name=f"diff_before_norm_{layer}",
        out_dir=out_dir,
        hover_labels=hover_labels,
    )

    create_pca_visualizations(
        hiddens=hidden_mean.view(-1, d),
        labels=expanded_labels,
        plot_name=f"mean_before_norm_{layer}",
        out_dir=out_dir,
        hover_labels=hover_labels,
    )

    create_pca_visualizations(
        hiddens=hiddens.view(-1, d),
        labels=duplicated_labels.view(-1),
        plot_name=f"all_before_norm_{layer}",
        out_dir=out_dir,
        hover_labels=None,
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
        hover_labels=hover_labels,
    )

    create_pca_visualizations(
        hiddens=hidden_mean.view(-1, d),
        labels=expanded_labels,
        plot_name=f"mean_after_norm_{layer}",
        out_dir=out_dir,
        hover_labels=hover_labels,
    )

    create_pca_visualizations(
        hiddens=normalized_hiddens.view(-1, d),
        labels=duplicated_labels.view(-1),
        plot_name=f"all_after_norm_{layer}",
        out_dir=out_dir,
        hover_labels=None,
    )
    # code to fit the CRC direction on the training data

"""Main training loop."""
import json
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Literal

import pandas as pd
import torch
from einops import rearrange, repeat
from simple_parsing import subgroups
from sklearn.cluster import KMeans

from elk.normalization.cluster_norm import split_clusters
from elk.utils.data_utils import PreparedData, prepare_data
from elk.utils.gpu_utils import get_device

from ..evaluation import Eval
from ..metrics import evaluate_preds, to_one_hot
from ..metrics.eval import LayerOutput
from ..run import LayerApplied, Run
from ..training.supervised import train_supervised
from ..utils.types import PromptEnsembling
from . import Classifier
from .ccs_reporter import CcsConfig, CcsReporter
from .common import FitterConfig
from .eigen_reporter import EigenFitter, EigenFitterConfig
from .multi_reporter import MultiReporter, ReporterWithInfo, SingleReporter

# For debugging, TODO: Remove later
torch.set_printoptions(threshold=5000)


def tensor_to_serializable(data):
    """
    Recursively converts tensors in the given data structure to a serializable format.
    """
    if isinstance(data, torch.Tensor):
        return data.tolist()  # Convert tensors to lists
    elif isinstance(data, dict):
        return {key: tensor_to_serializable(val) for key, val in data.items()}
    elif isinstance(data, list):
        return [tensor_to_serializable(val) for val in data]
    else:
        return data


def generate_html(json_data, output_file="clusters.html"):
    """
    Generates an HTML file that uses
    jquery.json-viewer (from CDN) to display the given JSON data.

    Parameters:
    json_data (dict): The JSON data to display.
    output_file (str): The filename for the generated HTML file.
    """

    # Convert the dictionary to a JSON-formatted string
    json_string = json.dumps(json_data, indent=4)

    # HTML content with jquery.json-viewer from CDN
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>JSON Viewer</title>
        <!-- jQuery from CDN -->
        <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>

        <!-- jquery.json-viewer JS and CSS from CDN -->
        <script src="https://cdn.jsdelivr.net/npm/jquery.json-viewer/json-viewer/jquery.json-viewer.js">
        </script>
        <link href="https://cdn.jsdelivr.net/npm/jquery.json-viewer/json-viewer/jquery.json-viewer.css" rel="stylesheet" type="text/css">
    </head>
    <body>
        <h2>JSON Data</h2>
        <pre id="json-renderer"></pre>

        <script>
            var jsonData = {json_string};
            $(document).ready(function() {{
                $('#json-renderer').jsonViewer(jsonData, {{collapsed: true}});
            }});
        </script>
    </body>
    </html>
    """  # noqa: E501

    # Write the HTML content to a file
    with open(output_file, "w") as file:
        file.write(html_content)


def flatten_text_questions(text_questions):
    # Initialize an empty list to store the flattened data
    flattened_text_questions = []

    # Loop through each sublist (representing the 'n' dimension)
    for sublist in text_questions:
        # Loop through each item in the sublist (representing the 'v' dimension)
        for item in sublist:
            # Append each item (which is a list of 'k' elements) to the flattened_list
            assert (
                item[0] == item[1]
            ), "First and second element should be the same, \
            only the pseudo label is different"

            # Store only the first element, since it is the same as the second
            # Only the pseudo label is different but that is appended later
            flattened_text_questions.append(item[0])
    return flattened_text_questions


# TODO: Move to utils
def get_clusters(
    x: torch.Tensor,
    labels: torch.Tensor,
    lm_preds: torch.Tensor,
    text_questions: list,
    num_clusters: int,
    cluster_algo: Literal["kmeans", None] = "kmeans",
) -> dict:
    n, v, k, d = x.shape

    # get rid of template dimensions,
    # since we are creating the clusters as a replacement
    x = x.view(n * v, k, d)
    labels = labels.repeat_interleave(v).flatten()
    lm_preds = lm_preds.view(n * v, k)

    x_averaged_over_choices = x.mean(dim=1)  # shape is (n * v, d)

    if cluster_algo == "kmeans":
        clustering_results = KMeans(
            n_clusters=num_clusters, random_state=0, n_init="auto"
        ).fit(x_averaged_over_choices.cpu().numpy())
    else:
        raise ValueError(f"Unknown cluster algo: {cluster_algo}")

    cluster_ids = clustering_results.labels_

    unique_clusters = list(set(cluster_ids.tolist()))
    print("unique_clusters", len(unique_clusters))

    clusters = {
        "train": {
            "hiddens": {},
            "labels": {},
            "lm_preds": {},
            "cluster_ids": {},
            "ids": {},
            "text_questions": {},
        },
        "test": {
            "hiddens": {},
            "labels": {},
            "lm_preds": {},
            "cluster_ids": {},
            "ids": {},
            "text_questions": {},
        },
    }
    text_questions = flatten_text_questions(text_questions)

    for unique_cluster_id in unique_clusters:
        cluster_data = []
        gt_data = []
        lm_pred_data = []
        # cluster ids and ids are collected for debugging,
        # cluster_ids should be all the same in a cluster
        cluster_ids_data = []
        # ids should be different
        ids_data = []

        # Gather data for the current cluster
        for idx, cluster_id in enumerate(cluster_ids):
            if cluster_id == unique_cluster_id:
                cluster_data.append(x[idx])
                gt_data.append(labels[idx])
                lm_pred_data.append(lm_preds[idx])
                cluster_ids_data.append(cluster_id.item())
                ids_data.append(idx)

        cluster = torch.stack(cluster_data, dim=0)

        # Divide the data in the middle
        # if there is just one element, add it to both train and test
        split_index = len(cluster_data) // 2 if len(cluster_data) > 1 else None
        clusters["train"]["hiddens"][unique_cluster_id] = cluster[:split_index]

        clusters["train"]["labels"][unique_cluster_id] = torch.stack(
            gt_data[:split_index], dim=0
        )
        clusters["train"]["lm_preds"][unique_cluster_id] = torch.stack(
            lm_pred_data[:split_index], dim=0
        )
        clusters["train"]["cluster_ids"][unique_cluster_id] = cluster_ids_data[
            :split_index
        ]
        clusters["train"]["ids"][unique_cluster_id] = ids_data[:split_index]

        selected_questions = [text_questions[idx] for idx in ids_data[:split_index]]
        clusters["train"]["text_questions"][unique_cluster_id] = selected_questions

        clusters["test"]["hiddens"][unique_cluster_id] = cluster[split_index:]
        clusters["test"]["labels"][unique_cluster_id] = torch.stack(
            gt_data[split_index:], dim=0
        )
        clusters["test"]["lm_preds"][unique_cluster_id] = torch.stack(
            lm_pred_data[split_index:], dim=0
        )
        clusters["test"]["cluster_ids"][unique_cluster_id] = cluster_ids_data[
            split_index:
        ]
        clusters["test"]["ids"][unique_cluster_id] = ids_data[split_index:]

        selected_questions = [text_questions[i] for i in ids_data[split_index:]]
        clusters["test"]["text_questions"][unique_cluster_id] = selected_questions

        assert set(cluster_ids_data[:split_index]) == {
            unique_cluster_id
        }, "cluster_ids should be all the same in a cluster"
        assert set(cluster_ids_data[split_index:]) == {
            unique_cluster_id
        }, "cluster_ids should be all the same in a cluster"
        assert len(set(ids_data[:split_index])) == len(
            ids_data[:split_index]
        ), "ids should be different"
        assert len(set(ids_data[split_index:])) == len(
            ids_data[split_index:]
        ), "ids should be different"

    return clusters


def evaluate_and_save_cluster(
    train_loss: float | None,
    reporter: SingleReporter | MultiReporter,
    clusters: dict,
    lr_models: list[Classifier],
    layer: int,
):
    row_bufs = defaultdict(list)
    layer_output = []

    for ds_name, value in clusters.items():
        val_cluster, val_labels, _ = (
            value["test"]["hiddens"],
            value["test"]["labels"],
            value["test"]["lm_preds"],
        )
        # train_cluster, _, _ = (
        #     value["train"]["hiddens"],
        #     value["train"]["labels"],
        #     value["train"]["lm_preds"],
        # )
        meta = {"dataset": ds_name, "layer": layer}

        val_neg, val_pos = split_clusters(val_cluster)
        val_credences_neg = reporter(val_neg)
        val_credences_pos = reporter(val_pos)
        val_credences = torch.stack(
            (val_credences_neg, val_credences_pos), dim=1
        )  # shape is (n, k) now, where k=2
        val_credences = val_credences.unsqueeze(1)  # now shape is (n, v, k), where v=1

        # train_pos, train_neg = split_clusters(train_cluster)
        # train_credences_neg = reporter(train_neg)
        # train_credences_pos = reporter(train_pos)
        # train_credences = torch.stack(
        # (train_credences_neg, train_credences_pos),
        # dim=1
        # )
        # train_credences = train_credences.unsqueeze(1)

        # Create a new dictionary to store the final result
        val_labels = torch.cat(list(val_labels.values()), dim=0)
        assert val_labels.dim() == 1, "Expected shape (n,)"

        layer_output.append(
            LayerOutput(
                val_gt=val_labels.detach(),
                val_credences=val_credences.detach(),
                meta=meta,
            )
        )
        PROMPT_ENSEMBLING = "prompt_ensembling"
        for prompt_ensembling in PromptEnsembling.all():
            row_bufs["eval"].append(
                {
                    **meta,
                    PROMPT_ENSEMBLING: prompt_ensembling.value,
                    **evaluate_preds(
                        val_labels, val_credences, prompt_ensembling
                    ).to_dict(),
                    "train_loss": train_loss,
                }
            )
            # row_bufs["train_eval"].append(
            #     {
            #         **meta,
            #         PROMPT_ENSEMBLING: prompt_ensembling.value,
            #         **evaluate_preds(
            #             train_gt,
            #             train_credences,
            #             prompt_ensembling
            #         ).to_dict(),
            #         "train_loss": train_loss,
            #     }
            # )

            # if val_lm_preds is not None:
            #     row_bufs["lm_eval"].append(
            #         {
            #             **meta,
            #             PROMPT_ENSEMBLING: prompt_ensembling.value,
            #             **evaluate_preds(
            #                 val_labels,
            #                 val_lm_preds,
            #                 prompt_ensembling
            #             ).to_dict(),
            #         }
            #     )

            # if train_lm_preds is not None:
            #     row_bufs["train_lm_eval"].append(
            #         {
            #             **meta,
            #             PROMPT_ENSEMBLING: prompt_ensembling.value,
            #             **evaluate_preds(
            #                 train_gt, train_lm_preds, prompt_ensembling
            #             ).to_dict(),
            #         }
            #     )
            # for lr_model_num, model in enumerate(lr_models):
            #     row_bufs["lr_eval"].append(
            #         {
            #             **meta,
            #             PROMPT_ENSEMBLING: prompt_ensembling.value,
            #             "inlp_iter": lr_model_num,
            #             **evaluate_preds(
            #                 val_labels, model(val_h), prompt_ensembling
            #             ).to_dict(),
            #         }
            #     )

    return LayerApplied(layer_output, {k: pd.DataFrame(v) for k, v in row_bufs.items()})


def evaluate_and_save(
    train_loss: float | None,
    reporter: SingleReporter | MultiReporter,
    train_dict: PreparedData,
    val_dict: PreparedData,
    lr_models: list[Classifier],
    layer: int,
):
    row_bufs = defaultdict(list)
    layer_output = []
    for ds_name in val_dict:
        val_h, val_gt, val_lm_preds = val_dict[ds_name]
        train_h, train_gt, train_lm_preds = train_dict[ds_name]
        meta = {"dataset": ds_name, "layer": layer}

        val_credences = reporter(val_h)
        train_credences = reporter(train_h)
        layer_output.append(
            LayerOutput(
                val_gt=val_gt.detach(),
                val_credences=val_credences.detach(),
                meta=meta,
            )
        )
        PROMPT_ENSEMBLING = "prompt_ensembling"
        for prompt_ensembling in PromptEnsembling.all():
            row_bufs["eval"].append(
                {
                    **meta,
                    PROMPT_ENSEMBLING: prompt_ensembling.value,
                    **evaluate_preds(
                        val_gt, val_credences, prompt_ensembling
                    ).to_dict(),
                    "train_loss": train_loss,
                }
            )

            row_bufs["train_eval"].append(
                {
                    **meta,
                    PROMPT_ENSEMBLING: prompt_ensembling.value,
                    **evaluate_preds(
                        train_gt, train_credences, prompt_ensembling
                    ).to_dict(),
                    "train_loss": train_loss,
                }
            )

            if val_lm_preds is not None:
                row_bufs["lm_eval"].append(
                    {
                        **meta,
                        PROMPT_ENSEMBLING: prompt_ensembling.value,
                        **evaluate_preds(
                            val_gt, val_lm_preds, prompt_ensembling
                        ).to_dict(),
                    }
                )

            if train_lm_preds is not None:
                row_bufs["train_lm_eval"].append(
                    {
                        **meta,
                        PROMPT_ENSEMBLING: prompt_ensembling.value,
                        **evaluate_preds(
                            train_gt, train_lm_preds, prompt_ensembling
                        ).to_dict(),
                    }
                )

            for lr_model_num, model in enumerate(lr_models):
                row_bufs["lr_eval"].append(
                    {
                        **meta,
                        PROMPT_ENSEMBLING: prompt_ensembling.value,
                        "inlp_iter": lr_model_num,
                        **evaluate_preds(
                            val_gt, model(val_h), prompt_ensembling
                        ).to_dict(),
                    }
                )

    return LayerApplied(layer_output, {k: pd.DataFrame(v) for k, v in row_bufs.items()})


@dataclass
class Elicit(Run):
    """Full specification of a reporter training run."""

    net: FitterConfig = subgroups(
        {"ccs": CcsConfig, "eigen": EigenFitterConfig}, default="eigen"  # type: ignore
    )
    """Config for building the reporter network."""

    supervised: Literal["none", "single", "inlp", "cv"] = "single"
    """Whether to train a supervised classifier, and if so, whether to use
    cross-validation. Defaults to "single", which means to train a single classifier
    on the training data. "cv" means to use cross-validation."""

    def make_eval(self, model, eval_dataset):
        assert self.out_dir is not None
        return Eval(
            data=replace(
                self.data,
                model=model,
                datasets=(eval_dataset,),
            ),
            source=self.out_dir,
            out_dir=self.out_dir / "transfer" / eval_dataset,
            num_gpus=self.num_gpus,
            min_gpu_mem=self.min_gpu_mem,
            skip_supervised=self.supervised == "none",
            prompt_indices=self.prompt_indices,
            concatenated_layer_offset=self.concatenated_layer_offset,
            # datasets isn't needed because it's immediately overwritten
            debug=self.debug,
            disable_cache=self.disable_cache,
        )

    # Create a separate function to handle the reporter training.
    def cluster_train_and_save_reporter(
        self, device, layer, out_dir, clusters, prompt_index=None
    ) -> ReporterWithInfo:
        train_loss = None
        if isinstance(self.net, CcsConfig):
            dataset_key = list(clusters.keys())[0]
            hiddens = clusters[dataset_key]["train"]["hiddens"]
            clusters[dataset_key]["train"]["labels"]

            d = hiddens[0].shape[-1]  # feature dimension are the same for all clusters
            reporter = CcsReporter(
                self.net,
                in_features=d,
                clusters_train=clusters[dataset_key]["train"],
                clusters_test=clusters[dataset_key]["test"],
                device=device,
            )
            train_loss = reporter.fit_by_clusters(hiddens)
            # iterate over hiddens
            # reporter.platt_scale_with_clusters(labels, hiddens)

        # Save reporter checkpoint to disk
        # TODO have to change this
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(reporter, out_dir / f"layer_{layer}.pt")

        return ReporterWithInfo(reporter, train_loss, prompt_index)

    # Create a separate function to handle the reporter training.
    def train_and_save_reporter(
        self, device, layer, out_dir, train_dict, prompt_index=None
    ) -> ReporterWithInfo:
        (first_train_h, train_gt, _), *rest = train_dict.values()  # TODO can remove?
        (_, v, k, d) = first_train_h.shape
        if not all(other_h.shape[-1] == d for other_h, _, _ in rest):
            raise ValueError("All datasets must have the same hidden state size")

        # For a while we did support datasets with different numbers of classes, but
        # we reverted this once we switched to ConceptEraser. There are a few options
        # for re-enabling it in the future but they are somewhat complex and it's not
        # clear that it's worth it.
        if not all(other_h.shape[-2] == k for other_h, _, _ in rest):
            raise ValueError("All datasets must have the same number of classes")

        train_loss = None
        if isinstance(self.net, CcsConfig):
            assert len(train_dict) == 1, "CCS only supports single-task training"
            # TODO: Do something else for clusters... Pass clusters
            (
                first_train_h,
                train_gt,
                _,
            ), *rest = train_dict.values()  # TODO can remove?
            (_, v, k, d) = first_train_h.shape
            reporter = CcsReporter(self.net, d, device=device)
            train_loss = reporter.fit(first_train_h)

            labels = repeat(to_one_hot(train_gt, k), "n k -> n v k", v=v)
            reporter.platt_scale(labels, first_train_h)

        elif isinstance(self.net, EigenFitterConfig):
            fitter = EigenFitter(
                self.net, d, num_classes=k, num_variants=v, device=device
            )

            hidden_list, label_list = [], []
            for ds_name, (train_h, train_gt, _) in train_dict.items():
                (_, v, _, _) = train_h.shape

                # Datasets can have different
                # numbers of variants, so we need to
                # flatten them here before concatenating
                hidden_list.append(rearrange(train_h, "n v k d -> (n v k) d"))
                label_list.append(
                    to_one_hot(repeat(train_gt, "n -> (n v)", v=v), k).flatten()
                )
                fitter.update(train_h)

            reporter = fitter.fit_streaming()
            reporter.platt_scale(
                torch.cat(label_list),
                torch.cat(hidden_list),
            )
        else:
            raise ValueError(f"Unknown reporter config type: {type(self.net)}")

        # Save reporter checkpoint to disk
        # TODO have to change this
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(reporter, out_dir / f"layer_{layer}.pt")

        return ReporterWithInfo(reporter, train_loss, prompt_index)

    def train_lr_model(self, train_dict, device, layer, out_dir) -> list[Classifier]:
        if self.supervised != "none":
            lr_models = train_supervised(
                train_dict,
                device=device,
                mode=self.supervised,
            )
            # make dir if not exists
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / f"layer_{layer}.pt", "wb") as file:
                torch.save(lr_models, file)
        else:
            lr_models = []

        return lr_models

    def apply_to_layer(
        self,
        layer: int,
        devices: list[str],
        world_size: int,
        probe_per_prompt: bool,
    ) -> LayerApplied:
        """Train a single reporter on a single layer."""
        assert self.out_dir is not None  # TODO this is really annoying, why can it be
        # None?

        self.make_reproducible(seed=self.net.seed + layer)
        device = get_device(devices, world_size)

        train_dict = prepare_data(self.datasets, device, layer, "train")
        val_dict = prepare_data(self.datasets, device, layer, "val")

        if self.net.norm == "cluster":
            clusters_by_dataset = {}
            for dataset_name, dataset in self.datasets:
                # TODO:
                # concatenate train and val hiddens and save the result in hiddens
                hiddens = train_dict[dataset_name][0]
                labels = train_dict[dataset_name][1]
                lm_preds = train_dict[dataset_name][2]
                text_questions = train_dict[dataset_name][3]

                _, v, _, _ = train_dict[dataset_name][0].shape

                if self.k_clusters is None:
                    self.k_clusters = v

                clusters = get_clusters(
                    hiddens, labels, lm_preds, text_questions, self.k_clusters
                )
                clusters_by_dataset[dataset_name] = clusters

                # print("train viz")
                # visualize_clusters(clusters["train"])
                # print("test viz")
                # visualize_clusters(clusters["test"])

                with open(self.out_dir / "clusters.json", "w"):
                    serialized_clusters = tensor_to_serializable(clusters)
                    json.dumps(serialized_clusters, indent=4)

            reporter_train_result = self.cluster_train_and_save_reporter(
                device, layer, self.out_dir / "reporters", clusters=clusters_by_dataset
            )

            maybe_multi_reporter = reporter_train_result.model
            train_loss = reporter_train_result.train_loss

            return evaluate_and_save_cluster(
                train_loss, maybe_multi_reporter, clusters_by_dataset, [], layer
            )
        else:
            if probe_per_prompt:
                (first_train_h, train_gt, _), *rest = train_dict.values()
                (_, v, k, d) = first_train_h.shape

                # self.prompt_indices being () actually means "all prompts"
                prompt_indices = (
                    self.prompt_indices if self.prompt_indices else range(v)
                )
                prompt_train_dicts = [
                    {
                        ds_name: (
                            train_h[:, [i], ...],
                            train_gt,
                            lm_preds[:, [i], ...] if lm_preds is not None else None,
                        )
                    }
                    for ds_name, (train_h, _, lm_preds) in train_dict.items()
                    for i, _ in enumerate(prompt_indices)
                ]

                results = []

                for prompt_index, prompt_train_dict in zip(
                    prompt_indices, prompt_train_dicts
                ):
                    assert prompt_index < 100  # format i as a 2 digit string
                    str_i = str(prompt_index).zfill(2)
                    base = self.out_dir / "reporters" / f"prompt_{str_i}"
                    reporters_path = base / "reporters"

                    reporter_train_result = self.train_and_save_reporter(
                        device, layer, reporters_path, prompt_train_dict, prompt_index
                    )
                    results.append(reporter_train_result)

                # it is called maybe_multi_reporter
                # because it might be a single reporter
                maybe_multi_reporter = MultiReporter(results)
                train_loss = maybe_multi_reporter.train_loss
            else:
                reporter_train_result = self.train_and_save_reporter(
                    device, layer, self.out_dir / "reporters", train_dict
                )

                maybe_multi_reporter = reporter_train_result.model
                train_loss = reporter_train_result.train_loss

            # TODO: Normalize by cluster for the lr_models too
            lr_models = self.train_lr_model(
                train_dict, device, layer, self.out_dir / "lr_models"
            )

            # TODO: Make it work for clusters
            return evaluate_and_save(
                train_loss, maybe_multi_reporter, train_dict, val_dict, lr_models, layer
            )

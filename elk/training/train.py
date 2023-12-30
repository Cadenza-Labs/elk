"""Main training loop."""

import pathlib
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Literal

import pandas as pd
import torch
from einops import rearrange, repeat
from matplotlib import pyplot as plt
from simple_parsing import subgroups
from sklearn.decomposition import PCA

from elk.training.burns_norm import BurnsNorm

from ..evaluation import Eval
from ..extraction import Extract
from ..metrics import evaluate_preds, to_one_hot
from ..metrics.eval import LayerOutput
from ..run import LayerApplied, PreparedData, Run
from ..training.supervised import train_supervised
from ..utils.types import PromptEnsembling
from . import Classifier
from .ccs_reporter import CcsConfig, CcsReporter
from .common import FitterConfig
from .eigen_reporter import EigenFitter, EigenFitterConfig
from .multi_reporter import MultiReporter, ReporterWithInfo, SingleReporter

DEEPMIND_REPRODUCTION = True


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

        if DEEPMIND_REPRODUCTION:
            train_h, train_gt = deepmind_reproduction(train_h, train_gt)

            val_h, val_gt = deepmind_reproduction(val_h, val_gt)

        def eval_all(reporter: SingleReporter | MultiReporter):
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

                if not DEEPMIND_REPRODUCTION:
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

        eval_all(reporter)

    return LayerApplied(layer_output, {k: pd.DataFrame(v) for k, v in row_bufs.items()})


def create_pca_visualizations(hiddens, labels, plot_name="pca_plot"):
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


def deepmind_reproduction(hiddens, gt_labels):
    assert hiddens.dim() == 4, "shape of hiddens has to be: (n, v, k, d)"
    n, v, k, d = hiddens.shape

    # Generate random indices for each template
    indices = torch.randperm(n)
    sample_size = n // 2
    indices_0 = indices[:sample_size]
    indices_1 = indices[sample_size:]

    # Select random samples from each template
    template_0_hiddens = hiddens[indices_0, 0, :, :]
    template_1_hiddens = hiddens[indices_1, 1, :, :]
    hiddens = torch.cat((template_0_hiddens, template_1_hiddens), dim=0)

    # Add "fake" template dimension to make it work with the rest of the code
    hiddens = torch.unsqueeze(hiddens, 1)  # (n, k, d) -> (n, 1, k, d)
    assert hiddens.shape == (n, 1, k, d), "shape of hiddens has to be: (n, 1, k, d)"

    gt_labels = torch.cat((gt_labels[indices_0], gt_labels[indices_1]), dim=0)
    assert gt_labels.shape == (n,), "shape of gt_labels has to be: (n,)"

    return hiddens, gt_labels


def pca_visualizations(layer, first_train_h, train_gt):
    n, v, k, d = first_train_h.shape
    flattened_hiddens = rearrange(first_train_h, "n v k d -> (n v k) d", v=v, k=k)
    expanded_labels = train_gt.repeat_interleave(v * k)

    create_pca_visualizations(
        hiddens=flattened_hiddens,
        labels=expanded_labels,
        plot_name=f"before_norm_{layer}",
    )

    # ... and after normalization
    norm = BurnsNorm()
    hiddens_neg, hiddens_pos = first_train_h.unbind(2)
    normalized_hiddens_neg = norm(hiddens_neg)
    normalized_hiddens_pos = norm(hiddens_pos)
    normalized_hiddens = torch.stack(
        (normalized_hiddens_neg, normalized_hiddens_pos), dim=2
    )
    flattened_normalized_hiddens = rearrange(
        normalized_hiddens, "n v k d -> (n v k) d", v=v, k=k
    )

    create_pca_visualizations(
        hiddens=flattened_normalized_hiddens,
        labels=expanded_labels,
        plot_name=f"after_norm_{layer}",
    )


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

    @staticmethod
    def default():
        return Elicit(
            data=Extract(
                model="<placeholder>",
                datasets=("<placeholder>",),
            )
        )

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
    def train_and_save_reporter(
        self, device, layer, out_dir, train_dict, prompt_index=None
    ) -> ReporterWithInfo:
        (first_train_h, train_gt, _), *rest = train_dict.values()  # TODO can remove?
        (_, v, k, d) = first_train_h.shape
        if not all(other_h.shape[-1] == d for other_h, _, _ in rest):
            raise ValueError("All datasets must have the same hidden state size")

        if DEEPMIND_REPRODUCTION:
            first_train_h, train_gt = deepmind_reproduction(first_train_h, train_gt)

        # For a while we did support datasets with different numbers of classes, but
        # we reverted this once we switched to ConceptEraser. There are a few options
        # for re-enabling it in the future but they are somewhat complex and it's not
        # clear that it's worth it.
        if not all(other_h.shape[-2] == k for other_h, _, _ in rest):
            raise ValueError("All datasets must have the same number of classes")

        train_loss = None
        if isinstance(self.net, CcsConfig):
            assert len(train_dict) == 1, "CCS only supports single-task training"
            reporter = CcsReporter(self.net, d, device=device, num_variants=v)
            train_loss = reporter.fit(first_train_h)
            reporter.training = False

            pca_visualizations(layer, first_train_h, train_gt)

            # Platt Scaling
            # labels = repeat(to_one_hot(train_gt, k),
            # "n k -> n v k", v=v)
            # reporter.platt_scale(labels, first_train_h)

        elif isinstance(self.net, EigenFitterConfig):
            fitter = EigenFitter(
                self.net, d, num_classes=k, num_variants=v, device=device
            )

            hidden_list, label_list = [], []
            for ds_name, (train_h, train_gt, _) in train_dict.items():
                (_, v, _, _) = train_h.shape

                # Datasets can have different numbers of variants, so we need to
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
        device = self.get_device(devices, world_size)

        train_dict = self.prepare_data(device, layer, "train")
        val_dict = self.prepare_data(device, layer, "val")

        if probe_per_prompt:
            (first_train_h, train_gt, _), *rest = train_dict.values()
            (_, v, k, d) = first_train_h.shape

            if DEEPMIND_REPRODUCTION:
                first_train_h, train_gt = deepmind_reproduction(first_train_h, train_gt)

            # self.prompt_indices being () actually means "all prompts"
            prompt_indices = self.prompt_indices if self.prompt_indices else range(v)
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

            # it is called maybe_multi_reporter because it might be a single reporter
            maybe_multi_reporter = MultiReporter(results)
            train_loss = maybe_multi_reporter.train_loss
        else:
            reporter_train_result = self.train_and_save_reporter(
                device, layer, self.out_dir / "reporters", train_dict
            )

            maybe_multi_reporter = reporter_train_result.model
            train_loss = reporter_train_result.train_loss

        lr_models = self.train_lr_model(
            train_dict, device, layer, self.out_dir / "lr_models"
        )

        return evaluate_and_save(
            train_loss, maybe_multi_reporter, train_dict, val_dict, lr_models, layer
        )

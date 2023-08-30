"""Main training loop."""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
import torch
from einops import rearrange, repeat
from simple_parsing import subgroups
from simple_parsing.helpers.serialization import save

from ..metrics import evaluate_preds, to_one_hot
from ..run import Run
from ..training.supervised import train_supervised
from ..utils.typing import assert_type
from .ccs_reporter import CcsConfig, CcsReporter
from .common import FitterConfig
from .eigen_reporter import EigenFitter, EigenFitterConfig


import os, csv
# Specify the filename

# Function to write to CSV
def write_to_csv(data, filename):
    # Check if the file exists
    file_exists = os.path.isfile(filename)
    
    # Open the CSV file in append mode
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # If file doesn't exist, write the header
        if not file_exists:
            writer.writerow(data.keys())

        # Write the rows
        writer.writerow(data.values())


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

    def create_models_dir(self, out_dir: Path):
        lr_dir = None
        lr_dir = out_dir / "lr_models"
        reporter_dir = out_dir / "reporters"

        lr_dir.mkdir(parents=True, exist_ok=True)
        reporter_dir.mkdir(parents=True, exist_ok=True)

        # Save the reporter config separately in the reporter directory
        # for convenient loading of reporters later.
        save(self.net, reporter_dir / "cfg.yaml", save_dc_types=True)

        return reporter_dir, lr_dir

    def apply_to_layer(
        self,
        layer: int,
        devices: list[str],
        world_size: int,
    ) -> dict[str, pd.DataFrame]:
        """Train a single reporter on a single layer."""

        self.make_reproducible(seed=self.net.seed + layer)
        device = self.get_device(devices, world_size)

        train_dict = self.prepare_data(device, layer, "train")
        val_dict = self.prepare_data(device, layer, "val")

        (first_train_h, train_gt, _), *rest = train_dict.values()
        (_, v, k, d) = first_train_h.shape

        ds_name = list(train_dict.keys())[0]

        filename = f'res.csv'

        def get_tpc(hiddens):
            U, S, V = torch.pca_lowrank(hiddens)
            return (hiddens @ V[..., :1]).squeeze(-1)

        # no norm
        def no_norm(x, gt):
            scores = get_tpc(x).argmax(dim=-1)
            y_true = repeat(gt, "n -> n v", v=v)
            return (scores == y_true).float().mean()

        # correct norm
        def norm(x):
            x_centered = x - x.mean(dim=0)
            std = torch.linalg.norm(x_centered, dim=0) / x_centered.shape[0] ** 0.5
            dims = tuple(range(1, std.dim()))
            avg_norm = std.mean(dim=dims, keepdim=True)
            return x_centered / avg_norm

        def correct_norm(x, gt):
            scores = norm(get_tpc(x)).argmax(dim=-1)
            y_true = repeat(gt, "n -> n v", v=v)
            return (scores == y_true).float().mean()
        # print('correct norm accuracy', five_sf(accuracy.item()))
        

        # incorrect norm
        def incorrect_norm(x, gt):
            tpc_wrong = rearrange(get_tpc(x), 'n v c -> (n v) c')
            scores = norm(tpc_wrong).argmax(dim=-1)
            y_true = repeat(gt, "n -> (n v)", v=v)
            return (scores == y_true).float().mean()
        
        
        

        # print("first_train_h.shape", first_train_h.shape)
        # print("train_gt.shape", train_gt.shape)
        # print("U.shape", U.shape)
        # print("S.shape", S.shape)
        # print("V.shape", V.shape)
        # print("ntrue.shape", ntrue.shape)
        # print("tpc.shape", tpc.shape)
        # print("scores.shape", scores.shape)

        if not all(other_h.shape[-1] == d for other_h, _, _ in rest):
            raise ValueError("All datasets must have the same hidden state size")

        # For a while we did support datasets with different numbers of classes, but
        # we reverted this once we switched to ConceptEraser. There are a few options
        # for re-enabling it in the future but they are somewhat complex and it's not
        # clear that it's worth it.
        if not all(other_h.shape[-2] == k for other_h, _, _ in rest):
            raise ValueError("All datasets must have the same number of classes")

        reporter_dir, lr_dir = self.create_models_dir(assert_type(Path, self.out_dir))
        train_loss = None

        if isinstance(self.net, CcsConfig):
            assert len(train_dict) == 1, "CCS only supports single-task training"

            reporter = CcsReporter(self.net, d, device=device, num_variants=v)
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
        torch.save(reporter, reporter_dir / f"layer_{layer}.pt")

        # Fit supervised logistic regression model
        if self.supervised != "none":
            lr_models = train_supervised(
                train_dict,
                device=device,
                mode=self.supervised,
            )
            with open(lr_dir / f"layer_{layer}.pt", "wb") as file:
                torch.save(lr_models, file)
        else:
            lr_models = []

        row_bufs = defaultdict(list)
        res = {}
        res['dataset'] = ds_name
        res['layer'] = layer
        for ds_name in val_dict:
            val_h, val_gt, val_lm_preds = val_dict[ds_name]
            train_h, train_gt, train_lm_preds = train_dict[ds_name]
            meta = {"dataset": ds_name, "layer": layer}

            val_credences = reporter(val_h)
            train_credences = reporter(train_h)
            res['no norm accuracy'] = no_norm(val_h, val_gt).item()
            res['correct norm accuracy'] = correct_norm(val_h, val_gt).item()
            res['incorrect norm accuracy'] = incorrect_norm(val_h, val_gt).item()
            for mode in ("none", "partial", "full"):
                eval_res = evaluate_preds(val_gt, val_credences, mode)
                res[f'eval_acc_{mode}'] = eval_res.accuracy.estimate
                row_bufs["eval"].append(
                    {
                        **meta,
                        "ensembling": mode,
                        **eval_res.to_dict(),
                        "train_loss": train_loss,
                    }
                )
                train_eval = evaluate_preds(train_gt, train_credences, mode)
                row_bufs["train_eval"].append(
                    {
                        **meta,
                        "ensembling": mode,
                        **train_eval.to_dict(),
                        "train_loss": train_loss,
                    }
                )

                # if val_lm_preds is not None:
                #     row_bufs["lm_eval"].append(
                #         {
                #             **meta,
                #             "ensembling": mode,
                #             **evaluate_preds(val_gt, val_lm_preds, mode).to_dict(),
                #         }
                #     )

                # if train_lm_preds is not None:
                #     row_bufs["train_lm_eval"].append(
                #         {
                #             **meta,
                #             "ensembling": mode,
                #             **evaluate_preds(train_gt, train_lm_preds, mode).to_dict(),
                #         }
                #     )

                # for i, model in enumerate(lr_models):
                #     row_bufs["lr_eval"].append(
                #         {
                #             **meta,
                #             "ensembling": mode,
                #             "inlp_iter": i,
                #             **evaluate_preds(val_gt, model(val_h), mode).to_dict(),
                #         }
                #     )

        # write res to csv
        write_to_csv(res, filename)

        return {k: pd.DataFrame(v) for k, v in row_bufs.items()}

"""Main training loop."""

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from einops import repeat
from simple_parsing import subgroups
from simple_parsing.helpers.serialization import save

from ..run import Run
from .ccs_reporter import CcsConfig
from .common import FitterConfig
from .eigen_reporter import EigenFitterConfig


def plot(x, y, z, model_name, ds_name, layer):
    expt_name = f"{model_name}_{ds_name}_L{layer}"
    # plot y against x
    plt.scatter(x.detach().cpu().numpy(), y.detach().cpu().numpy())
    # add linear regression

    m, b = np.polyfit(x.detach().cpu().numpy(), y.detach().cpu().numpy(), 1)
    label4 = "(4): cos(pseudo_dir, probe_dir)^2"
    plt.plot(x.detach().cpu().numpy(), m * x.detach().cpu().numpy() + b, label=label4)

    # add linear regression
    m, b = np.polyfit(z.detach().cpu().numpy(), y.detach().cpu().numpy(), 1)
    label5 = "(5): ‖Φ⁺ - Φ⁺′‖²  / ‖Φ⁺ - Φ⁺″‖²"

    plt.scatter(z.detach().cpu().numpy(), y.detach().cpu().numpy())
    plt.plot(z.detach().cpu().numpy(), m * z.detach().cpu().numpy() + b, label=label5)

    # add legend
    plt.legend()

    # add title
    plt.title(f"{model_name} | {ds_name} | layer {layer}")

    # add axis labels
    plt.xlabel("x")
    plt.ylabel("[accuracy w/t-wise norm]-[accuracy wo/t-wise norm]")

    plt.savefig(f"expt/{expt_name}.png")
    # clear
    plt.clf()


# Specify the filename


# Function to write to CSV
def write_to_csv(data, filename):
    # Check if the file exists
    file_exists = os.path.isfile(filename)

    # Open the CSV file in append mode
    with open(filename, "a", newline="") as csvfile:
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

    out_dir_expt: Path = Path("./out")

    def __post_init__(self):
        # make dir
        if not os.path.exists(self.out_dir_expt):
            self.out_dir_expt.mkdir(parents=True, exist_ok=True)

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
        model_name = self.data.model.replace("/", "_")

        expt_name = f"{model_name}_{ds_name}_L{layer}"

        to_save = {}

        def expt_2_3(hiddens: torch.Tensor):
            # hiddens is n v c d
            h_mean = hiddens.mean(dim=0)  # v c d
            assert h_mean.shape == (v, k, d)
            pseudolabel_directions = h_mean[:, 1, :] - h_mean[:, 0, :]  # v d
            assert pseudolabel_directions.shape == (v, d)
            pseudo_t_wise_avg = pseudolabel_directions.mean(dim=0, keepdim=True)  # 1 d
            assert pseudo_t_wise_avg.shape == (1, d)
            pseudolabel_directions_avg = torch.concatenate(
                [pseudolabel_directions, pseudo_t_wise_avg]
            )  # v+1 d
            assert pseudolabel_directions_avg.shape == (v + 1, d)

            norms = torch.linalg.norm(pseudolabel_directions_avg, dim=-1)
            cosine_similarities = F.cosine_similarity(
                pseudolabel_directions_avg.unsqueeze(0),
                pseudolabel_directions_avg.unsqueeze(1),
                dim=-1,
            )
            # pretty print pairwise similarity
            # rich.print(norms)
            # rich.print(pd.DataFrame(cosine_similarities.detach().cpu().numpy()))
            to_save["2_3_norms"] = norms.detach().cpu().numpy().tolist()
            to_save["2_3_cosine_similarities"] = (
                cosine_similarities.detach().cpu().numpy().tolist()
            )
            return cosine_similarities

        expt_2_3(first_train_h)

        class TPCProbe:
            def __init__(self, probe_direction):
                self.probe_direction = probe_direction

            @classmethod
            def train(cls, train_hiddens: torch.Tensor, correct_norm=True):
                n, v, c, d = train_hiddens.shape
                wrong = not correct_norm
                x_pos, x_neg = norm(train_hiddens[..., 1, :], wrong=wrong), norm(
                    train_hiddens[..., 0, :], wrong=wrong
                )  # n v d
                assert x_pos.shape == x_neg.shape == (n, v, d)
                C = x_pos - x_neg  # n v d
                assert C.shape == (n, v, d)
                _, _, V = torch.pca_lowrank(C.flatten(end_dim=-2))  # n v d
                return cls(V[..., :1])

            def apply(self, hiddens: torch.Tensor, correct_norm=True):
                n, v, c, d = hiddens.shape
                wrong = not correct_norm
                hiddens.mean(dim=0)
                x_pos, x_neg = norm(hiddens[..., 1, :], wrong=wrong), norm(
                    hiddens[..., 0, :], wrong=wrong
                )  # n v d
                assert x_pos.shape == x_neg.shape == (n, v, d)

                C = x_pos - x_neg  # n v d
                assert C.shape == (n, v, d)
                logits = (C @ self.probe_direction).squeeze(-1)  # n v
                assert logits.shape == (n, v)
                return logits

        def norm(x, wrong=False):
            assert x.dim() == 3, "x must be n v d"
            n, v, d = x.shape
            mean_dims = (0, 1) if wrong else 0  # if wrong select n and v else just n
            x_centered = x - x.mean(dim=mean_dims)  # n v d  |  (n v) d
            assert x_centered.shape == (n, v, d)
            std = (
                torch.linalg.norm(x_centered, dim=0) / x_centered.shape[0] ** 0.5
            )  # v d  | d
            assert std.shape == (v, d)
            std_dims = (0, 1) if wrong else 1  # (1)  |  ()
            avg_norm = std.mean(dim=std_dims)  # if wrong select d and v else just d
            if wrong:
                assert avg_norm.shape == ()
            else:
                assert avg_norm.shape == (v,)
            if wrong:
                normed = x_centered / avg_norm
            else:
                normed = x_centered / avg_norm[:, None]
            assert normed.shape == (n, v, d)
            return normed  # n v d

        def get_acc(x_train, x_val, gt, correct_norm=True):
            probe = TPCProbe.train(x_train, correct_norm=correct_norm)
            scores = probe.apply(x_val, correct_norm=correct_norm).gt(0)
            y_true = repeat(gt, "n -> n v", v=v)
            return scores == y_true

        def expt_6_svd(hiddens: torch.Tensor):
            @dataclass
            class Expt:
                hiddens: torch.Tensor
                # fn is a function that takes a tensor and returns a tensor
                fn: Callable[[torch.Tensor], torch.Tensor]
                name: str

                def run(self):
                    x_pos = hiddens[:, :, 1, :]
                    x_neg = hiddens[:, :, 0, :]
                    pseudolabel_directions = (self.fn(x_pos) - self.fn(x_neg)).mean(
                        dim=0
                    )
                    assert pseudolabel_directions.shape == (v, d)
                    U, S, V = torch.linalg.svd(pseudolabel_directions)
                    for prefix, M in zip("USV", (U, S, V)):
                        to_save[f"6_svd_{self.name}_{prefix}"] = (
                            M.detach().cpu().numpy().tolist()
                        )

                    breakpoint()

            Expt(hiddens, lambda x: x, "no_norm").run()
            Expt(hiddens, lambda x: norm(x, wrong=False), "correct_norm").run()
            Expt(hiddens, lambda x: norm(x, wrong=True), "wrong_norm").run()

        def expt_4_5(train_hiddens, val_hiddens):  # n v c d
            x_pos, x_neg = norm(train_hiddens[:, :, 1, :], wrong=True), norm(
                train_hiddens[:, :, 0, :], wrong=True
            )  # n v d
            tpc_probe_wrong = TPCProbe.train(train_hiddens, correct_norm=False)
            wrong_credences = tpc_probe_wrong.apply(val_hiddens, correct_norm=False)
            wrong_y = (
                wrong_credences.gt(0).bool() == repeat(val_gt, "n -> n v", v=v).bool()
            )  # n v
            wrong_acc = wrong_y.float().mean(dim=0)  # v

            tpc_probe_correct = TPCProbe.train(train_hiddens, correct_norm=True)
            right_credences = tpc_probe_correct.apply(
                val_hiddens, correct_norm=True
            )  # n v
            y_true = repeat(val_gt, "n -> n v", v=v)  # n v
            right_acc = (right_credences.gt(0) == y_true).float().mean(dim=0)

            # [accuracy on this template with template - wise normalization]-
            # [accuracy on this template without template-wise normalization]
            y = right_acc - wrong_acc
            to_save["4_5_y"] = y.detach().cpu().numpy().tolist()

            new_pseudolabel_directions = (x_pos - x_neg).mean(dim=0)  # v d
            # x_i = [1 - cosine of the angle between the
            # pseudolabel direction and the probe direction]
            x = F.cosine_similarity(
                new_pseudolabel_directions, tpc_probe_wrong.probe_direction.squeeze(-1)
            ).pow(2)
            assert x.shape == (v,)
            to_save["4_cos_pseudo_probe_x"] = x.detach().cpu().numpy().tolist()

            # ‖ phi ^ +-phi ^ +'‖^2 / ||phi^+ - phi^+''||^2
            Φ_a = (x_pos + x_neg) / 2 + new_pseudolabel_directions / 2
            Φ_b = (x_pos + x_neg) / 2
            # z = torch.linalg.vector_norm(x_pos - Φ_a, dim=(0,2)) /
            # torch.linalg.vector_norm(x_pos - Φ_b, dim=(0,2))
            ratios = torch.linalg.vector_norm(
                x_pos - Φ_a, dim=2
            ) / torch.linalg.vector_norm(x_pos - Φ_b, dim=2)
            z = ratios.mean(dim=0)
            # plot y against z

            to_save["5_ratios_z"] = z.detach().cpu().numpy().tolist()

            # plot(x, y, z, model_name, ds_name, layer)

        res = {"dataset": ds_name, "layer": layer, "model": model_name}
        for ds_name in val_dict:
            val_h, val_gt, val_lm_preds = val_dict[ds_name]
            train_h, train_gt, train_lm_preds = train_dict[ds_name]
            expt_4_5(first_train_h, val_h)
            expt_6_svd(first_train_h)

            # val_credences = reporter(val_h)
            # train_credences = reporter(train_h)
            to_save["correct_norm_accuracy"] = (
                get_acc(train_h, val_h, val_gt, correct_norm=True).float().tolist()
            )
            to_save["incorrect_norm_accuracy"] = (
                get_acc(train_h, val_h, val_gt, correct_norm=False).float().tolist()
            )
            res["correct_norm_accuracy"] = to_save["correct_norm_accuracy"]
            res["incorrect_norm_accuracy"] = to_save["incorrect_norm_accuracy"]

        # make res into df row and append if exists else create
        # breakpoint()
        def save_acc_df():
            df = pd.DataFrame(res, index=[0])
            by_layer_filename = self.out_dir_expt / "accs_by_norm.csv"
            by_layer_filename.parent.mkdir(parents=True, exist_ok=True)
            if os.path.exists(by_layer_filename):
                # check if dataset, layer, model exists
                existing_df = pd.read_csv(by_layer_filename)
                # find row with same dataset, layer, model
                existing_row = existing_df[
                    (existing_df["dataset"] == ds_name)
                    & (existing_df["layer"] == layer)
                    & (existing_df["model"] == model_name)
                ]
                if existing_row.empty:
                    df.to_csv(by_layer_filename, mode="a", header=False, index=False)
                else:
                    print(f"{ds_name}, {layer}, {model_name} already exists in csv")
            else:
                df.to_csv(by_layer_filename, index=False)
                df.columns = list(res.keys())

        # write to_save
        with open(self.out_dir_expt / f"{expt_name}.json", "w") as f:
            json.dump(to_save, f, indent=4)

        return {}

"""Main training loop."""

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rich
import json
import torch
import torch.nn.functional as F
from einops import repeat
from simple_parsing import subgroups
from simple_parsing.helpers.serialization import save

from .ccs_reporter import CcsConfig
from .common import FitterConfig
from .eigen_reporter import EigenFitterConfig
from ..run import Run


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
        model_name = self.data.model.replace("/", "_")

        expt_name = f'{model_name}_{ds_name}_layer_{layer}'

        filename = f'{expt_name}.csv'

        to_save = {}

        def expt_2_3(hiddens: torch.Tensor):
            # hiddens is n v c d
            h_mean = hiddens.mean(dim=0)  # v c d
            pseudolabel_directions = h_mean[:, 0, :] - h_mean[:, 1, :]  # v d
            pseudo_t_wise_avg = pseudolabel_directions.mean(dim=0, keepdim=True)  # 1 d
            pseudolabel_directions_avg = torch.concatenate([pseudolabel_directions, pseudo_t_wise_avg])  # v+1 d

            norms = torch.linalg.norm(pseudolabel_directions_avg, dim=-1)
            cosine_similarities = F.cosine_similarity(pseudolabel_directions_avg.unsqueeze(0),
                                                      pseudolabel_directions_avg.unsqueeze(1),
                                                      dim=-1)
            # pretty print pairwise similarity
            rich.print(norms)
            rich.print(pd.DataFrame(cosine_similarities.detach().cpu().numpy()))
            to_save['2_3_norms'] = norms.detach().cpu().numpy().tolist()
            to_save['2_3_cosine_similarities'] = cosine_similarities.detach().cpu().numpy().tolist()
            return cosine_similarities

        e23 = expt_2_3(first_train_h)

        def tpc_probe(hiddens, wrong=False):  # n v c d
            # what is probe direction?
            x_pos, x_neg = norm(hiddens[..., 1, :], wrong=wrong), norm(hiddens[..., 0, :], wrong=wrong)  # n v d
            C = x_pos - x_neg  # n v d
            U, S, V = torch.pca_lowrank(C.flatten(end_dim=-2))  # n v d
            return (C @ V[..., :1]).squeeze(-1)  # n v

        def norm(x, wrong=False):
            assert x.dim() == 3, "x must be n v d"
            mean_dims = (0, 1) if wrong else 0  # if wrong select n and v else just n
            x_centered = x - x.mean(dim=mean_dims)  # n v d  |  (n v) d
            std = torch.linalg.norm(x_centered, dim=0) / x_centered.shape[0] ** 0.5  # v d  | d
            std_dims = (0, 1) if wrong else 1  # (1)  |  ()
            avg_norm = std.mean(dim=std_dims, keepdim=True)  # if wrong select d and v else just d
            return x_centered / avg_norm  # n v d

        def get_acc(x, gt, probe, wrong=False):
            scores = probe(x, wrong=wrong).gt(0)
            y_true = repeat(gt, "n -> n v", v=v)
            return scores == y_true

        def expt_4_5(train_hiddens, val_hiddens, ds_name):  # n v c d
            x_pos, x_neg = norm(train_hiddens[:, :, 1, :], wrong=True), norm(train_hiddens[:, :, 0, :],
                                                                             wrong=True)  # n v d

            C_train = x_pos - x_neg  # n v d
            U, S, V = torch.pca_lowrank(C_train.flatten(end_dim=-2))
            probe_direction = V[..., 0]  # d

            x_pos_val, x_neg_val = norm(val_hiddens[..., 0, :], wrong=True), norm(val_hiddens[..., 1, :],
                                                                                  wrong=True)  # n v d
            C_val = x_pos_val - x_neg_val
            wrong_credences = (C_val @ V[..., :1]).squeeze(-1)  # n v
            wrong_y = wrong_credences.gt(0).bool() == repeat(val_gt, "n -> n v", v=v).bool()  # n v
            wrong_acc = wrong_y.float().mean(dim=0)  # v

            right_credences = tpc_probe(train_hiddens, wrong=False)  # n v
            y_true = repeat(val_gt, "n -> n v", v=v)  # n v
            right_acc = (right_credences.gt(0) == y_true).float().mean(dim=0)
            # [accuracy on this template with template - wise normalization]-
            # [accuracy on this template without template-wise normalization]
            y = right_acc - wrong_acc
            to_save['4_5_y'] = y.detach().cpu().numpy().tolist()

            new_pseudolabel_directions = (x_neg - x_pos).mean(dim=0)  # v d
            # x_i = [1 - cosine of the angle between the pseudolabel direction and the probe direction]
            x = 1 - F.cosine_similarity(new_pseudolabel_directions, probe_direction.unsqueeze(0))
            to_save['4_cos_pseudo_probe_x'] = x.detach().cpu().numpy().tolist()

            # plot y against x
            plt.scatter(x.detach().cpu().numpy(), y.detach().cpu().numpy())
            # add linear regression

            m, b = np.polyfit(x.detach().cpu().numpy(), y.detach().cpu().numpy(), 1)
            label4 = '(4): 1 - cos(pseudo_dir, probe_dir)'
            plt.plot(x.detach().cpu().numpy(), m * x.detach().cpu().numpy() + b, label=label4)
            # make dir
            if not os.path.exists('expt'):
                os.makedirs('expt')

            # ‖ phi ^ +-phi ^ +'‖^2 / ||phi^+ - phi^+''||^2
            Φ_a = (x_pos + x_neg) / 2 + new_pseudolabel_directions / 2
            Φ_b = (x_pos + x_neg) / 2
            # z = torch.linalg.vector_norm(x_pos - Φ_a, dim=(0,2)) / torch.linalg.vector_norm(x_pos - Φ_b,
            #                                                                                    dim=(0,2))
            ratios = torch.linalg.vector_norm(x_pos - Φ_a, dim=2) / torch.linalg.vector_norm(x_pos - Φ_b,
                                                                                               dim=2)
            z = ratios.mean(dim=0)
            # plot y against z
            plt.scatter(z.detach().cpu().numpy(), y.detach().cpu().numpy())
            to_save['5_ratios_z'] = z.detach().cpu().numpy().tolist()
            # add linear regression
            m, b = np.polyfit(z.detach().cpu().numpy(), y.detach().cpu().numpy(), 1)
            label5 = '(5): ‖Φ⁺ - Φ⁺′‖²  / ‖Φ⁺ - Φ⁺″‖²'
            plt.plot(z.detach().cpu().numpy(), m * z.detach().cpu().numpy() + b, label=label5)

            # add legend
            plt.legend()

            # add title
            plt.title(f'{model_name} | {ds_name} | layer {layer}')

            # add axis labels
            plt.xlabel('x')
            plt.ylabel('[accuracy w/t-wise norm]-[accuracy wo/t-wise norm]')

            plt.savefig(f'expt/{expt_name}.png')

        res = {'dataset': ds_name, 'layer': layer}
        for ds_name in val_dict:
            val_h, val_gt, val_lm_preds = val_dict[ds_name]
            train_h, train_gt, train_lm_preds = train_dict[ds_name]
            # meta = {"dataset": ds_name, "layer": layer}
            expt_4_5(first_train_h, val_h, ds_name)

            # val_credences = reporter(val_h)
            # train_credences = reporter(train_h)
            to_save['correct_norm_accuracy'] = get_acc(val_h, val_gt, probe=tpc_probe, wrong=False).float().mean().item()
            to_save['incorrect_norm_accuracy'] = get_acc(val_h, val_gt, probe=tpc_probe, wrong=True).float().mean().item()

        #  write res to csv
        write_to_csv(res, filename)

        # write to_save
        with open(f'expt/{expt_name}.json', 'w') as f:
            json.dump(to_save, f, indent=4)

        return {}

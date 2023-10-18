from abc import ABC
from dataclasses import dataclass

import torch
from kmeans_pytorch import kmeans
from simple_parsing.helpers import Serializable

from elk.extraction import Extract, extract
from elk.utils.data_utils import prepare_data
from elk.utils.gpu_utils import get_device

from .utils import (
    select_usable_devices,
)

use_cuda = torch.cuda.is_available()
dtype = torch.float32 if use_cuda else torch.float64
device_id = "cuda:0" if use_cuda else "cpu"


@dataclass
class Kmeans(ABC, Serializable):
    data: Extract
    num_gpus: int = 4
    world_size: int = 1
    min_gpu_mem: int | None = None
    prompt_indices: tuple[int, ...] = ()

    def get_clusters(self, X: torch.Tensor, cluster_ids: list) -> dict:
        """
        Create a dictionary where each key is a unique cluster ID and
        the associated value is a list of
        data points (from X) that belong to that cluster.

        Parameters:
        - X (torch.Tensor): A tensor containing the data points.
        - cluster_ids (list): A list containing the cluster ID
        for each data point in X.

        Returns:
        - dict: A dictionary with unique cluster IDs as keys
        and lists of data points as values.
        """
        clusters = {}
        for unique_cluster_id in list(set(cluster_ids)):
            clusters[unique_cluster_id] = []

        for idx, cluster_id in enumerate(cluster_ids):
            clusters[cluster_id].append(X[idx])

        return clusters

    def apply(self, X, k=10):
        cluster_ids, cluster_centers = kmeans(
            X=X,
            num_clusters=k,
            distance="euclidean",
            device=torch.device("cuda:0"),  # TODO: make it work for more than one GPU
        )
        self.get_clusters(X, cluster_ids.tolist())

    def execute(self):
        datasets = [
            extract(
                cfg,
                disable_cache=False,
                num_gpus=self.num_gpus,
            )
            for cfg in self.data.explode()
        ]
        devices = select_usable_devices(self.num_gpus, min_memory=self.min_gpu_mem)

        layer = 1
        device = get_device(devices, self.world_size)
        train_dict = prepare_data(datasets, device, layer, "train")
        prepare_data(datasets, device, layer, "val")

        hiddens = train_dict["imdb"][0]
        averaged_over_choices = hiddens.mean(dim=2)
        n, v, d = averaged_over_choices.shape
        reshaped = averaged_over_choices.view(n * v, d)
        cluster_data = self.apply(X=reshaped, k=v)
        print(cluster_data)

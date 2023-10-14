from abc import ABC
from dataclasses import dataclass

from simple_parsing.helpers import Serializable

from elk.extraction import Extract, extract
from elk.utils.data_utils import prepare_data
from elk.utils.gpu_utils import get_device

from .utils import (
    select_usable_devices,
)


@dataclass
class Kmeans(ABC, Serializable):
    data: Extract
    num_gpus: int = 4
    world_size: int = 1
    min_gpu_mem: int | None = None
    prompt_indices: tuple[int, ...] = ()

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
        val_dict = prepare_data(datasets, device, layer, "val")

        print("train_dict", train_dict)
        print("val_dict", val_dict)


# X = np.array([[1, 2], [1, 4], [1, 0],
#               [10, 2], [10, 4], [10, 0]])
# kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X)

# kmeans.labels_
# kmeans.predict([[0, 0], [12, 3]])
# kmeans.cluster_centers_

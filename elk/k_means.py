from abc import ABC
from dataclasses import dataclass

from simple_parsing.helpers import Serializable

from elk.extraction import Extract, extract


@dataclass
class Kmeans(ABC, Serializable):
    data: Extract

    def execute(self):
        datasets = [
            extract(
                cfg,
                disable_cache=False,
                num_gpus=4,
            )
            for cfg in self.data.explode()
        ]

        print("datasets", datasets)


# X = np.array([[1, 2], [1, 4], [1, 0],
#               [10, 2], [10, 4], [10, 0]])
# kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X)

# kmeans.labels_
# kmeans.predict([[0, 0], [12, 3]])
# kmeans.cluster_centers_

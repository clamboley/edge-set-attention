from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import InMemoryDataset
from torch_geometric.datasets import MNISTSuperpixels

from src.ESA.processing import crop_or_pad, edge_adjacency, edge_features


def get_trainset() -> InMemoryDataset:
    """Load the training set."""
    return MNISTSuperpixels(root="data", train=True)


def get_testset() -> InMemoryDataset:
    """Load the test set."""
    return MNISTSuperpixels(root="data", train=False)


class ContrastiveMNIST(Dataset):
    """Contrastive dataset where an example is returned with N positive samples."""

    def __init__(
        self,
        data: InMemoryDataset,
        n_views: int = 2,
        block_size: int | None = None,
    ) -> None:
        """Initialize dataset."""
        if n_views < 1:
            msg = "n_views must be greater than 0"
            raise ValueError(msg)

        if block_size is None:
            block_size = max(graph.edge_index.size(1) for graph in data)

        self.examples = data
        self.block_size = block_size
        self.n_positives = n_views - 1

        self.label_to_indices = defaultdict(list)
        for index, label in enumerate(data.y.tolist()):
            self.label_to_indices[label].append(index)

    def __len__(self) -> int:
        """Length of the dataset."""
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple:
        """Retrieve a list of similar items from the dataset.

        Select random examples with the same label as the anchor (example idx), excluding
        the anchor itself. Return features, mask and label for the anchor and positive samples.

        Args:
            idx (int): index of the anchor example.

        Returns:
            tuple: (features, mask, label) for the anchor and positive samples.
        """
        anchor = self.examples[idx]
        label = int(anchor.y)

        positive_indices = []
        if self.n_positives > 0:
            rng = np.random.default_rng()

            positive_indices = self.label_to_indices.get(label, [])
            positive_indices = list(filter(lambda x: x != idx, positive_indices))

            # Choose n_positives samples
            positive_indices = list(
                rng.choice(positive_indices, size=self.n_positives, replace=False),
            )

        group_data = [anchor] + [self.examples[i] for i in positive_indices]
        group_features, group_mask = [], []
        for data in group_data:
            # Using both node feature and position
            data.x = torch.cat([data.x, data.pos], dim=1)

            features = edge_features(data)
            mask = edge_adjacency(data)

            features, mask = crop_or_pad(features, mask, self.block_size)
            group_features.append(features)
            group_mask.append(mask)

        return torch.stack(group_features), torch.stack(group_mask), label

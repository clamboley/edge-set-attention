"""ESA data processing pipeline.

Using a simple graph dataset (MUTAG), transform it into an ESA compatible format.
"""

import torch
from torch.nn import functional as F  # noqa: N812
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset

from ESA.processing import edge_features, edge_adjacency, crop_or_pad


class GraphDataset(Dataset):
    """MUTAG dataset transformed into fixed-size edge features format."""

    def __init__(self, block_size: int) -> None:
        super().__init__()

        self.dataset = TUDataset(root="data", name="MUTAG")
        self.block_size = block_size

    def __len__(self) -> int:
        """Get the number of graphs in the dataset."""
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a graph and transform it into edge features and mask."""
        data = self.dataset[index]

        features = edge_features(data)
        adjacency_matrix = edge_adjacency(data)

        features, adjacency_matrix = crop_or_pad(features, adjacency_matrix)

        return features, adjacency_matrix, data.y

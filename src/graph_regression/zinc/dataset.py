from torch.utils.data import Dataset
from torch_geometric.datasets import ZINC

from src.ESA.processing import crop_or_pad, edge_adjacency, edge_features


class EdgeOrientedZINC(Dataset):
    """Dataset for edge-oriented ZINC dataset."""

    def __init__(
        self,
        split: str,
        block_size: int | None = None,
    ) -> None:
        """Initialize ZINC dataset."""
        self.data = ZINC(root="data\\zinc", split=split)

        self.block_size = block_size
        if self.block_size is None:
            self.block_size = max(graph.edge_index.size(1) for graph in self.data)

    def __len__(self) -> int:
        """Length of the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple:
        """Get item from dataset and preprocess it into its edge-oriented representation.

        NOTE: We could preprocess the whole dataset at once, but this would require a lot of memory.
           A preprocessed dataset would take up around 100 times more memory than the original
           dataset, mostly due to the adjacency matrices. Instead, we preprocess each item on the
           fly. This is slower, but more memory-efficient.

        Args:
            idx (int): Index of the item to get.

        Returns:
            tuple: Tuple containing the features, mask, and label of the item.
        """
        data = self.data[idx]

        # Extract edge features and adjacency matrix
        features = edge_features(data)
        mask = edge_adjacency(data)

        features, mask = crop_or_pad(features, mask, self.block_size)

        return features, mask, data.y

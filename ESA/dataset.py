"""ESA data processing pipeline.

Using a simple graph dataset (MUTAG), transform it into an ESA compatible format.
"""

import torch
from torch.nn import functional as F  # noqa: N812
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset


def edge_adjacency(data: Data) -> torch.Tensor:
    """Find the edge adjacency matrix from a graph.

    An edge is adjacent to another edge if they share a node,
    wether as a source or as a target, there is no difference.
    """
    edge_index = data.edge_index
    num_nodes = edge_index.size(1)

    source_nodes = edge_index[0]
    target_nodes = edge_index[1]

    exp_src = source_nodes.unsqueeze(1).expand((-1, num_nodes))
    exp_trg = target_nodes.unsqueeze(1).expand((-1, num_nodes))

    # Using sources and targets for the case of a directed graph
    src_adj = exp_src == exp_src.T
    trg_adj = exp_trg == exp_trg.T
    cross_adj = (exp_src == exp_trg.T) + (exp_trg == exp_src.T)

    return src_adj + trg_adj + cross_adj


def edge_features(data: Data) -> torch.Tensor:
    """Create edge features from a graph.

    Edge features are the concatenation of the source node features,
    target node features, and the edge features.
    """
    edge_index = data.edge_index
    edge_attr = data.edge_attr
    node_features = data.x

    source_features = node_features[edge_index[0]]
    target_features = node_features[edge_index[1]]

    # Concatenate source, target, and edge attributes
    return torch.cat([source_features, target_features, edge_attr], dim=1)


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

        if (num_edges := features.size(0)) > self.block_size:
            # Crop to block_size
            features = features[: self.block_size]
            adjacency_matrix = adjacency_matrix[: self.block_size, : self.block_size]

        elif num_edges < self.block_size:
            pad_size = self.block_size - num_edges

            # Pad to block_size
            features = F.pad(features, (0, 0, 0, pad_size), "constant", 0)

            # Pad to block_size on the bottom and right side
            adjacency_matrix = F.pad(adjacency_matrix, (0, pad_size, 0, pad_size), "constant", 0)

        return features, adjacency_matrix, data.y

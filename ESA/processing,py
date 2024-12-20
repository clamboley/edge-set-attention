"""ESA data processing.

Define functions to process a pytorch_geometric graph and
tranform it to ESA-compatible edge features and mask.
"""

import torch
from torch.nn import functional as F  # noqa: N812
from torch_geometric.data import Data


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

    features = [source_features, target_features]
    if edge_attr is not None:
        features += [edge_attr]

    # Concatenate source, target, and edge attributes
    return torch.cat(features, dim=1)


def crop_or_pad(
    features: torch.Tensor,
    mask: torch.Tensor,
    target_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Change edge features and mask size by cropping or padding.

    Args:
        features (torch.Tensor): Edge features of shape (num_edges, features_dim)
        mask (torch.Tensor): Edge adjacency mask of shape (num_edges, num_edges)
        target_length (int): Target length to crop or pad to.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Features and mask of the right size.
    """
    if (num_edges := features.size(0)) > target_length:
        # Crop to target_length
        features = features[:target_length]
        mask = mask[:target_length, :target_length]

    elif num_edges < target_length:
        pad_size = target_length - num_edges

        # Pad to target_length
        features = F.pad(features, (0, 0, 0, pad_size), "constant", 0)
        mask = F.pad(mask, (0, pad_size, 0, pad_size), "constant", 0)

    return features, mask

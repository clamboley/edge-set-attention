import pytest
import torch
from torch_geometric.data import Data

from src.ESA.processing import edge_adjacency, edge_features


@pytest.fixture
def mock_graph() -> Data:
    """Return a mock ESAConfig object."""
    # Define 3-dimensional node features for 6 nodes
    x = torch.tensor(
        [
            [1.0, 0.0, 0.5],
            [0.5, 1.0, 0.2],
            [0.3, 0.8, 0.7],
            [0.2, 0.4, 0.9],
            [0.6, 0.1, 0.8],
            [0.9, 0.3, 0.4],
        ],
        dtype=torch.float,
    )

    # Define edges (source, target)
    edge_index = torch.tensor(
        [
            [0, 0, 1, 2, 2, 3, 3, 3, 4, 5],
            [1, 2, 2, 1, 3, 2, 4, 5, 5, 4],
        ],
        dtype=torch.long,
    )

    # Define 2-dimensional edge features for the 10 edges
    edge_attr = torch.tensor(
        [
            [0.1, 0.2],
            [0.2, 0.3],
            [0.3, 0.4],
            [0.4, 0.5],
            [0.5, 0.6],
            [0.6, 0.7],
            [0.7, 0.8],
            [0.8, 0.9],
            [0.9, 1.0],
            [1.0, 1.1],
        ],
        dtype=torch.float,
    )

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=0)


def test_edge_adjacency(mock_graph: Data) -> None:
    """Test the edge adjacency matrix computation."""
    adj_matrix = edge_adjacency(mock_graph)

    num_edges = mock_graph.edge_index.size(1)
    assert adj_matrix.size() == (num_edges, num_edges)
    assert torch.all(adj_matrix.diag() == 1), "Diagonal of adjacency matrix should be all ones."

    assert adj_matrix[0, 1] == 1, "Edges 0 and 1 share a source node"
    assert adj_matrix[0, 3] == 1, "Edges 0 and 3 share a target node"
    assert adj_matrix[4, 2] == 1, "Edge 4 source node is edge 2 target node"
    assert adj_matrix[0, 7] == 0, "Edges 0 and 7 are not adjacent"


def test_edge_features(mock_graph: Data) -> None:
    """Test the edge feature computation."""
    features = edge_features(mock_graph)

    # Validate the output shape (should be [num_edges, concatenated_feature_dim])
    num_edges = mock_graph.edge_index.size(1)
    node_feature_dim = mock_graph.x.size(1)
    edge_feature_dim = mock_graph.edge_attr.size(1)
    expected_dim = 2 * node_feature_dim + edge_feature_dim
    assert features.size() == (num_edges, expected_dim), "Wrong features dimensions"

    # Check correctness of a specific edge's features
    edge_index = mock_graph.edge_index
    node_features = mock_graph.x
    edge_attr = mock_graph.edge_attr

    for i in range(num_edges):
        source_idx = edge_index[0, i]
        target_idx = edge_index[1, i]
        expected_features = torch.cat(
            [
                node_features[source_idx],
                node_features[target_idx],
                edge_attr[i],
            ],
        )
        assert torch.allclose(features[i], expected_features)

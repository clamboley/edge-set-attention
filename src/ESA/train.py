import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader

from ESA.dataset import GraphDataset
from ESA.model import ESA, ESAConfig


class ModelForBinaryClassification(nn.Module):
    """Model with an added classification head."""

    def __init__(self, base_model: nn.Module, feature_dim: int) -> None:
        super().__init__()
        self.base_model = base_model
        self.classifier = nn.Linear(feature_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Use the model for binary classification."""
        embeddings = self.base_model(x, mask)
        return self.classifier(embeddings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model parameters
    parser.add_argument(
        "--layers",
        type=str,
        default="SMMMMS",
        help="A list of self 'S' and masked 'M' attention blocks for the encoder.",
    )
    parser.add_argument(
        "--n-head",
        type=int,
        default=6,
        help="Number of heads in the multi-head attention blocks.",
    )
    parser.add_argument(
        "--n-embd",
        type=int,
        default=18,  # node_features * 2 + edge_features = 18 dimensions
        help="Dimension of hidden states embeddings.",
    )
    parser.add_argument(
        "--pool-seeds",
        type=int,
        default=16,
        help="Number of seed vectors for the pooling module.",
    )
    parser.add_argument(
        "--pool-layers",
        type=int,
        default=2,
        help="Number of self attention blocks in the pooling module.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout for regularization.",
    )
    # Dataset parameters
    parser.add_argument(
        "--block-size",
        type=int,
        default=64,
        help="Maximum number of edges in a graph (Context length).",
    )
    # Training parameters
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for training and evaluation.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Train for this many epochs",
    )
    args = parser.parse_args()

    dataset = GraphDataset(block_size=args.block_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = ESAConfig(
        layers=args.layers,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        seeds=args.pool_seeds,
        pool_layers=args.pool_layers,
    )

    encoder = ESA(config)
    model = ModelForBinaryClassification(encoder, feature_dim=args.n_embd)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train for one epoch
    model.train()
    step = 0
    for epoch in range(args.epochs):
        for i, batch in enumerate(dataloader):
            step += 1
            edge_representations, edge_adjacency_matrix, labels = (
                batch[0].to(device),
                batch[1].to(device),
                batch[2].to(device),
            )

            logits = model(edge_representations, edge_adjacency_matrix)

            loss = criterion(logits, labels.float())

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging
            if (step) % 10 == 0:
                print(f"Epoch {epoch} ({i+1:3}/{len(dataloader)}): Loss: {loss.item():.4f}")

import torch
from torch import nn

from src.ESA.model import ESA, ESAConfig


class RegressionESA(torch.nn.Module):
    """ESA model for regression tasks."""

    def __init__(self, config: ESAConfig) -> None:
        """Initialize the RegressionESA model."""
        super().__init__()
        self.esa = ESA(config)
        self.regression_head = nn.Linear(config.n_embd, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass of the RegressionESA model."""
        z = self.esa(x, mask)
        return self.regression_head(z)

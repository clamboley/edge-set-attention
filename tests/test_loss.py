import pytest
import torch

from src.loss import ContrastiveLoss


def create_dummy_data(
    batch_size: int,
    num_views: int,
    feature_dim: int,
    *,
    labels: bool = False,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create dummy data for testing."""
    torch.manual_seed(seed)
    features = torch.randn(batch_size, num_views, feature_dim)
    if labels:
        labels = torch.randint(0, 10, (batch_size,))
    return features, labels


def test_input_dim() -> None:
    """Test error handling with too few input dimensions."""
    loss_fn = ContrastiveLoss()
    features = torch.randn(4, 2)
    with pytest.raises(ValueError, match="Expected input with at least 3 dimensions"):
        loss_fn(features)


def test_labels_mask() -> None:
    """Test error handling with both labels and mask."""
    loss_fn = ContrastiveLoss()
    features, labels = create_dummy_data(4, 2, 8, labels=True)
    mask = torch.eye(4)
    with pytest.raises(ValueError, match="Cannot define both 'labels' and 'mask'."):
        loss_fn(features, labels, mask)

def test_unsupervised() -> None:
    """Test unsupervised mode."""
    loss_fn = ContrastiveLoss()
    features, _ = create_dummy_data(4, 2, 8)
    loss = loss_fn(features)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0

def test_supervised_all() -> None:
    """Test supervised mode with contrast_mode="all"."""
    loss_fn = ContrastiveLoss(contrast_mode="all")
    features, labels = create_dummy_data(4, 2, 8, labels=True)
    loss = loss_fn(features, labels)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0

def test_supervised_one() -> None:
    """Test supervised mode with contrast_mode="one"."""
    loss_fn = ContrastiveLoss(contrast_mode="one")
    features, labels = create_dummy_data(4, 2, 8, labels=True)
    loss = loss_fn(features, labels)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0

def test_custom_mask() -> None:
    """Test supervised mode with custom mask."""
    loss_fn = ContrastiveLoss()
    features, _ = create_dummy_data(4, 2, 8)
    mask = torch.eye(4)
    loss = loss_fn(features, mask=mask)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0

def test_all_same_label() -> None:
    """Test edge case where all samples have the same label."""
    loss_fn = ContrastiveLoss()
    features, _ = create_dummy_data(4, 2, 8)
    labels = torch.tensor([0, 0, 0, 0])
    loss = loss_fn(features, labels)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0

def test_single_pair() -> None:
    """Test edge case with a single pair."""
    loss_fn = ContrastiveLoss()
    features, labels = create_dummy_data(1, 2, 8, labels=True)
    loss = loss_fn(features, labels)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0

def test_single_view() -> None:
    """Test edge case with a single view."""
    loss_fn = ContrastiveLoss()
    features, labels = create_dummy_data(4, 1, 8, labels=True)
    loss = loss_fn(features, labels)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0

def test_single_example() -> None:
    """Test edge case with a single sample."""
    loss_fn = ContrastiveLoss()
    features, labels = create_dummy_data(1, 1, 8, labels=True)
    loss = loss_fn(features, labels)
    # The loss should be 0.0 since there is only one sample.
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert torch.isclose(loss, torch.tensor(0.0))

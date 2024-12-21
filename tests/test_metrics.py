import torch

from src.metrics import mean_average_precision, pairwise_cosine


def test_map_simple_example() -> None:
    """Test mean_average_precision in a simple case."""
    similarities = torch.tensor(
        [
            [1.0, 0.9, 0.3, 0.2],
            [0.9, 1.0, 0.4, 0.1],
            [0.3, 0.4, 1.0, 0.8],
            [0.2, 0.1, 0.8, 1.0],
        ],
    )
    labels = torch.tensor([0, 0, 1, 1])

    mean_ap, precision = mean_average_precision(similarities, labels)

    assert torch.isclose(mean_ap, torch.tensor(1.0), atol=1e-6)
    assert torch.isclose(precision, torch.tensor(1.0), atol=1e-6)


def test_map_no_relevant_item() -> None:
    """Test mean_average_precision where a label appear only one time."""
    similarities = torch.tensor(
        [
            [1.0, 0.9, 0.1, 0.2],
            [0.9, 1.0, 0.1, 0.2],
            [0.1, 0.1, 1.0, 0.8],
            [0.2, 0.2, 0.8, 1.0],
        ],
    )
    labels = torch.tensor([0, 0, 1, 2])

    mean_ap, precision = mean_average_precision(similarities, labels)

    assert torch.isnan(mean_ap)
    assert torch.isnan(precision)


def test_map_single_sample() -> None:
    """Test mean_average_precision with a dingle sample as input."""
    similarities = torch.tensor([[1.0]])
    labels = torch.tensor([0])

    mean_ap, precision = mean_average_precision(similarities, labels)

    assert torch.isnan(mean_ap)
    assert torch.isnan(precision)


def test_map_larger_example() -> None:
    """Test mean_average_precision with a bigger input."""
    similarities = torch.randn(4096, 4096)
    labels = torch.randint(0, 16, (4096,))

    mean_ap, precision = mean_average_precision(similarities, labels)

    assert mean_ap >= 0.0
    assert mean_ap <= 1.0
    assert precision >= 0.0
    assert precision <= 1.0


def test_pairwise_cosine_orthogonal() -> None:
    """Test pairwise_cosine with orthogonal vectors."""
    embs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    cosine_sim = pairwise_cosine(embs)

    expected = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    assert torch.allclose(cosine_sim, expected, atol=1e-6)

def test_pairwise_cosine_identical() -> None:
    """Test pairwise_cosine with identical vectors."""
    embs = torch.tensor([[1.0, 1.0], [1.0, 1.0]])

    cosine_sim = pairwise_cosine(embs)

    expected = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    assert torch.allclose(cosine_sim, expected, atol=1e-6)

def test_pairwise_cosine_larger() -> None:
    """Test pairwise_cosine with a bigger input."""
    embs = torch.randn(1024, 512)

    cosine_sim = pairwise_cosine(embs)

    assert cosine_sim.size() == (1024, 1024)

def test_pairwise_cosine_safe_mode() -> None:
    """Test pairwise_cosine with division by zero in safe mode."""
    embs = torch.tensor([[0.0, 0.0], [1.0, 1.0]])

    cosine_sim = pairwise_cosine(embs, safe=True)

    expected = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    assert torch.allclose(cosine_sim, expected, atol=1e-6)

def test_pairwise_cosine_unsafe_mode() -> None:
    """Test pairwise_cosine with division by zero in unsafe mode."""
    embs = torch.tensor([[0.0, 0.0], [1.0, 1.0]])

    cosine_sim = pairwise_cosine(embs, safe=False)

    assert cosine_sim[0][0] == 0.0
    assert cosine_sim[0][1].isnan()
    assert cosine_sim[1][0].isnan()
    assert cosine_sim[1][1] == 0.0

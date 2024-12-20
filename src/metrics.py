import torch


def mean_average_precision(
    similarities: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor]:
    """Compute the Retrieval Mean Average Precision.

    Args:
        similarities (torch.Tensor): Pairwise similarities of size (samples, samples).
        labels (torch.Tensor): Of size (samples,)

    Returns:
        torch.Tensor: mAP@K as a float Tensor.
    """
    label_counts = torch.bincount(labels)  # Find k for every label

    # We only keep top max_k elements to reduce memory usage
    max_k = label_counts.max().item() - 1
    _, sorted_indices = similarities.topk(k=max_k, dim=-1)
    sorted_labels = labels[sorted_indices]

    correct_neighbors = torch.zeros_like(labels, dtype=torch.float32, device=similarities.device)
    average_precision = torch.zeros_like(labels, dtype=torch.float32, device=similarities.device)
    for i in range(len(similarities)):
        label = labels[i]
        k = label_counts[label] - 1
        true_positions = sorted_labels[i, :k] == label

        correct_neighbors[i] = true_positions.sum() / k
        precision_at_k = torch.cumsum(true_positions, dim=0) / (
            torch.arange(k, device=similarities.device) + 1
        )
        average_precision[i] = torch.sum(precision_at_k[true_positions]) / k

    precision = correct_neighbors.mean()
    mean_average_precision = average_precision.mean()

    return mean_average_precision, precision


def pairwise_cosine(embs: torch.Tensor, *, safe: bool = True) -> torch.Tensor:
    """Compute cosine similarity between embeddings.

    Args:
        embs (torch.Tensor): Embeddings Tensor of size (N, D).
        safe (bool, optional): Safe mode to avoid division by zero.

    Returns:
        torch.Tensor: Cosine similarity Tensor of size (N, N).
    """
    # Normalize first.
    if safe:
        embs_div = torch.linalg.vector_norm(embs, dim=1, keepdim=True)
        embs = embs / torch.clamp(embs_div, min=1e-8)
    else:
        embs = embs / embs.norm(dim=1, keepdim=True)

    # Scalar product of normalized vectors gives cosine similarity.
    cosine_similarities = torch.mm(embs, embs.T)
    cosine_similarities.fill_diagonal_(0)
    return cosine_similarities

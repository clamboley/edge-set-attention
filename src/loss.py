import torch
from torch import nn


class ContrastiveLoss(nn.Module):
    """Contrastive Loss for learning representations.

    Supports Unsupervised (SimCLR) and Supervised (SupCon) mode.
      - https://arxiv.org/abs/2002.05709
      - https://arxiv.org/abs/2004.11362
    """

    FEATURES_DIM: int = 3
    EPSILON: float = 1e-6

    def __init__(self, temperature: float = 0.1, contrast_mode: str = "all") -> None:
        """Initialize the contrastive loss.

        Args:
            temperature (float): Smaller temperature benefits training more than higher ones, but
                extremely low temperatures are harder to train due to numerical instability.
            contrast_mode (str): Can be either "one" or "all". Default is "all".
                - "one": Only the first view of each sample is used as the anchor.
                - "all": All views of each sample are used as anchors.
        """
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor = None,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute the supervised contrative loss.

        If both `labels` and `mask` are None, it degenerates to SimCLR unsupervised loss.

        Args:
            features (torch.Tensor): hidden vector of shape (bsz, n_views, ...).
            labels (torch.Tensor): ground truth of shape (bsz).
            mask (torch.Tensor): contrastive mask of shape (bsz, bsz), mask[i, j] = 1
                if sample j has the same label as sample i. Can be asymetric.

        Returns:
            A loss scalar.
        """
        if features.dim() < self.FEATURES_DIM:
            msg = f"Expected input with at least 3 dimensions, got shape {features.shape}."
            raise ValueError(msg)

        batch_size = features.size(0)
        contrast_count = features.size(1)
        if features.dim() > self.FEATURES_DIM:
            features = features.view(batch_size, contrast_count, -1)

        if labels is not None and mask is not None:
            msg = "Cannot define both 'labels' and 'mask'."
            raise ValueError(msg)

        # Degenerates to SimCLR
        if labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.bool, device=features.device)

        # Use labels to create a mask
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.size(0) != batch_size:
                msg = f"Num of labels doesn't match batch size: {labels.size(0)} != {batch_size}."
                raise ValueError(msg)
            mask = torch.eq(labels, labels.T).to(features.device)

        # Use a mask directly
        else:
            mask = mask.bool().to(features.device)

        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            msg = f"Unknown contrast mode: {self.contrast_mode}."
            raise ValueError(msg)

        # Compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature,
        )

        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(features.device),
            0,
        )
        mask = mask * logits_mask

        # Log probs
        exp_logits = torch.exp(logits) * logits_mask
        log_probs = logits - torch.log(exp_logits.sum(1, keepdim=True) + self.EPSILON)

        # Mean of log likelihood over positives
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < self.EPSILON, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_probs).sum(1) / mask_pos_pairs

        # Loss
        loss = -mean_log_prob_pos
        return loss.view(anchor_count, batch_size).mean()

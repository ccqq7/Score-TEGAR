"""
Asymmetric Loss for Multi-Label Classification (ASL).

Based on: "Asymmetric Loss For Multi-Label Classification" (ICCV 2021)
Paper: https://arxiv.org/abs/2009.14119

ASL addresses the positive-negative imbalance in multi-label classification
by using different focusing parameters for positive and negative samples.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss (ASL) for multi-label classification.

    Uses different gamma values for positive (gamma_pos) and negative (gamma_neg)
    samples, plus asymmetric clipping to suppress easy negatives.

    Args:
        gamma_neg: Focusing parameter for negative samples. Higher = more
                   suppression of easy negatives. Default: 4.
        gamma_pos: Focusing parameter for positive samples. 0 = standard CE
                   for positives (recommended). Default: 0.
        clip: Probability margin for hard-thresholding easy negatives.
              Set to 0 to disable. Default: 0.05.
        reduction: 'mean' or 'none'. Default: 'mean'.
    """

    def __init__(
        self,
        gamma_neg: float = 4,
        gamma_pos: float = 0,
        clip: float = 0.05,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw model output (before sigmoid), shape (B, C).
            targets: Binary targets, shape (B, C).

        Returns:
            Loss tensor (scalar if reduction='mean', shape (B, C) if 'none').
        """
        # ASL contains sigmoid/log operations that are numerically unsafe in
        # float16 for CLIP-scale logits.  In particular, sigmoid values below
        # the smallest float16 subnormal become zero and ``log(0)`` produces
        # inf; the focusing weight can then turn that into NaN.  Keep model
        # forward/backward under AMP, but evaluate the loss itself in FP32.
        loss_logits = logits.float()
        loss_targets = targets.float()
        xs_pos = torch.sigmoid(loss_logits)
        xs_neg = 1.0 - xs_pos

        # Asymmetric Clipping: shift negative probabilities to suppress easy negatives
        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        # Binary cross-entropy (split into positive and negative parts)
        loss_pos = loss_targets * torch.log(xs_pos.clamp(min=1e-8))
        loss_neg = (1.0 - loss_targets) * torch.log(xs_neg.clamp(min=1e-8))
        loss = loss_pos + loss_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * loss_targets  # probability for positive samples
            pt1 = xs_neg * (1.0 - loss_targets)  # probability for negative samples
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * loss_targets + self.gamma_neg * (1.0 - loss_targets)
            one_sided_w = torch.pow(1.0 - pt, one_sided_gamma)
            loss = loss * one_sided_w

        if self.reduction == "mean":
            return -loss.mean()
        return -loss


class AsymmetricLossWithMask(nn.Module):
    """ASL with label mask support for zero-shot learning.

    Applies ASL only on masked (seen) labels and averages correctly.
    """

    def __init__(
        self,
        gamma_neg: float = 4,
        gamma_pos: float = 0,
        clip: float = 0.05,
    ) -> None:
        super().__init__()
        self.asl = AsymmetricLoss(
            gamma_neg=gamma_neg,
            gamma_pos=gamma_pos,
            clip=clip,
            reduction="none",
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        label_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, C) raw logits.
            targets: (B, C) binary targets.
            label_mask: (C,) binary mask (1=seen, 0=unseen).

        Returns:
            Scalar loss averaged over seen labels and batch.
        """
        loss_per_label = self.asl(logits, targets)  # (B, C)
        masked_loss = loss_per_label * label_mask.unsqueeze(0)
        return masked_loss.sum() / (label_mask.sum().clamp_min(1.0) * logits.size(0))

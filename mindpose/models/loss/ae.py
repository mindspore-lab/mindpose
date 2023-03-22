from typing import Optional

import mindspore as ms
import mindspore.ops as ops
from mindspore import Tensor

from ...register import register
from .loss import Loss


@register("loss", extra_name="ae")
class AELoss(Loss):
    """Associative embedding loss. Or called `Grouping loss`. Based on
    `"End-to-End Learning for Joint Detection and Grouping"
    <https://arxiv.org/abs/1611.05424>`_.

    Args:
        tag_per_joint: Whether each of the joint has its own coordinate encoding.
            Default: True
        reduction: Type of the reduction to be applied to loss. The optional value
            are "mean", "sum" and "none". Default: "mean"

    Inputs:
        | pred: Predicted tags. In shape [N, K, H, W] if tag_per_joint is True; in
            shape [N, H, W] otherwise. Where K stands for the number of joints.
        | target: Ground truth of tag mask. In shape [N, M, K, 2] if tag_per_joint is
            True; in shape [N, M, 2] otherwise. Where M stands for number of instances.

    Outputs:
        | loss: Loss tensor contains the push loss and the pull loss.
    """

    def __init__(
        self, tag_per_joint: bool = True, reduction: Optional[str] = "mean"
    ) -> None:
        super().__init__(reduction=reduction)
        self.tag_per_joint = tag_per_joint
        self.eps = 0.01

    def construct(self, pred: Tensor, target: Tensor) -> Tensor:
        if not self.tag_per_joint:
            # insert the dimension K=1
            pred = pred[:, None, ...]
            target = target[..., None, :]

        N, K, H, W = pred.shape
        M = target.shape[1]

        # convert the index [N, M, K, 2] to mask [N, M, K, H, W]
        target_mask = ops.zeros((N, M, K, H * W), pred.dtype)
        update = ops.cast(target[..., 1:2], pred.dtype)
        target_mask = ops.tensor_scatter_elements(
            target_mask, target[..., 0:1], update, axis=3
        )
        target_mask = ops.reshape(target_mask, (N, M, K, H, W))

        target_mask_bool = ops.cast(target_mask, ms.bool_)

        pred = pred[:, None, ...]
        pred = ops.masked_fill(pred, ~target_mask_bool, 0.0)

        # calculate the reference embedding for each instance
        k_n = target_mask.sum(axis=(2, 3, 4))
        h_n = pred.sum(axis=(2, 3, 4)) / (k_n + self.eps)

        # calculate the pull loss
        diff = (h_n[..., None, None, None] - pred) * target_mask
        pull_loss = (diff**2).sum(axis=(2, 3, 4)) / (k_n + self.eps)
        mask = ops.cast(k_n > 0, diff.dtype)
        m = mask.sum(axis=1)
        pull_loss = pull_loss.sum(axis=1) / (m + self.eps)

        # calculate the push loss
        A = ops.broadcast_to(h_n[..., None], (N, M, M))
        B = ops.transpose(A, (0, 2, 1))
        diff = A - B
        push_loss = ops.exp(-(diff**2))
        # invalid h_n will not contribute to the loss
        diff_mask = ops.broadcast_to(mask[..., None], (N, M, M))
        diff_mask = diff_mask * ops.transpose(diff_mask, (0, 2, 1))
        push_loss *= diff_mask
        push_loss = push_loss.sum(axis=(1, 2))
        # remove the diagonal value
        push_loss -= m
        push_loss = 0.5 * push_loss / (m * (m - 1) + self.eps)

        push_loss = self.get_loss(push_loss)
        pull_loss = self.get_loss(pull_loss)
        return ops.stack([push_loss, pull_loss])

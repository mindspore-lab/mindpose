from typing import Tuple, Union

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
        sigma: Sigma value in push loss. Default: 1.0
        reduce: Whether reduce the loss into the single value. Default: False

    Inputs:
        | pred: Predicted tags. In shape [N, K, H, W]. Where K stands for the
            number of joints.
        | target: Ground truth of tag mask. In shape [N, M, K, H, W]. Where M stands
            for number of instances.

    Outputs:
        | loss: Loss value if reduce is True; Otherwise return the tuple of pull and
            push loss
    """

    def __init__(self, sigma: float = 1.0, reduce: bool = False) -> None:
        super().__init__()
        self.sigma = sigma
        self.reduce = reduce
        self.eps = 0.001

    def construct(
        self, pred: Tensor, target: Tensor
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        N, M, K, _, _ = target.shape

        target_bool = ops.cast(target, ms.bool_)
        target = ops.cast(target, pred.dtype)

        pred = pred[:, None, ...]
        pred = ops.masked_fill(pred, ~target_bool, 0.0)

        # calculate the reference embedding for each instance
        h_n = pred.sum(axis=(2, 3, 4)) / (target.sum(axis=(2, 3, 4)) + self.eps)

        # calculate the pull loss
        diff = (h_n[..., None, None, None] - pred) * target
        pull_loss = (diff**2).sum(axis=(1, 2, 3, 4)) / (
            target.sum(axis=(1, 2, 3, 4)) + self.eps
        )

        # calculate the push loss
        A = ops.broadcast_to(h_n[..., None], (N, M, M))
        B = ops.transpose(A, (0, 2, 1))
        diff = A - B
        push_loss = ops.exp(-(diff**2) / (self.sigma**2 * 2))
        # invalid h_n will not contribute to the loss
        mask = ops.cast(target.sum(axis=(2, 3, 4)) > 0, diff.dtype)
        diff_mask = ops.broadcast_to(mask[..., None], (N, M, M))
        diff_mask = diff_mask * ops.transpose(diff_mask, (0, 2, 1))
        push_loss *= diff_mask
        push_loss = push_loss.sum(axis=(1, 2))
        # remove the diagonal value
        m = mask.sum(axis=1)
        push_loss -= m
        push_loss = 0.5 * push_loss / (m * (m - 1) + self.eps)

        # multiply the pull loss by K such that two loss has roughly same scale
        pull_loss *= K

        if self.reduce:
            push_loss = push_loss.mean()
            pull_loss = pull_loss.mean()
            return push_loss + pull_loss

        return pull_loss, push_loss

from typing import Tuple, Union

import mindspore as ms
import mindspore.ops as ops
from mindspore import Tensor

from ...register import register
from .loss import Loss


@register("loss", extra_name="ae")
class AELoss(Loss):
    """Associative embedding loss. Or called `Grouping loss`. Implemented in vectorized way.

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

    def construct(
        self, pred: Tensor, target: Tensor
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        N, M, K, _, _ = target.shape

        target = ops.cast(target, ms.bool_)

        pred = pred[:, None, ...]
        pred = ops.masked_fill(pred, ~target, 0.0)

        # calculate the reference embedding for each instance
        h_n = pred.sum(axis=(2, 3, 4)) / K

        # calculate the pull loss
        diff = (h_n[..., None, None, None] - pred) * target
        pull_loss = (diff**2).sum(axis=(1, 2, 3, 4)) / (M * K)

        # calculate the push loss
        A = ops.broadcast_to(h_n[..., None], (N, M, M))
        B = ops.transpose(A, (0, 2, 1))
        diff = A - B
        push_loss = (
            ops.exp(-1 / 2 / self.sigma**2 * diff**2).sum(axis=(1, 2)) / M**2
        )

        if self.reduce:
            return push_loss.mean() + pull_loss.mean()

        return pull_loss, push_loss

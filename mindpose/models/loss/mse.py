from typing import Optional

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

from ...register import register
from .loss import Loss


@register("loss", extra_name="joint_mse")
class JointsMSELoss(Loss):
    """Joint Mean square error loss.
    It is the MSE loss of heatmaps with extra weight for different channel.

    Args:
        use_target_weight: Use extra weight in loss calculation

    Inputs:
        | pred: Predictions, in shape [N, C, H, W]
        | target: Ground truth, in shape [N, C, H, W]
        | target_weight: Loss weight, in shape [N, C]

    Outputs:
        | loss: Loss value
    """

    def __init__(self, use_target_weight: bool = False) -> None:
        super().__init__()
        self.use_target_weight = use_target_weight
        if self.use_target_weight:
            self.criterion = nn.MSELoss(reduction="none")
        else:
            self.criterion = nn.MSELoss(reduction="mean")

    def construct(
        self, pred: Tensor, target: Tensor, target_weight: Optional[Tensor] = None
    ) -> Tensor:
        if self.use_target_weight:
            loss = self.criterion(pred, target)
            loss = target_weight[..., None, None] * loss
            loss = ops.mean(loss)
        else:
            loss = self.criterion(pred, target)
        return loss


@register("loss", extra_name="joint_mse_with_mask")
class JointsMSELossWithMask(Loss):
    """Joint Mean square error loss with mask.
    Mask-out position will not contribute to the loss.

    Inputs:
        | pred: Predictions, in shape [N, C, H, W]
        | target: Ground truth, in shape [N, C, H, W]
        | mask: Ground truth Mask, in shape [N, H, W]

    Outputs:
        | loss: Loss value
    """

    def __init__(self) -> None:
        super().__init__()
        self.criterion = nn.MSELoss(reduction="none")

    def construct(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        loss = self.criterion(pred, target)
        loss = mask[:, None, :, :] * loss
        loss = ops.mean(loss)
        return loss

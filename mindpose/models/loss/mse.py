from typing import Optional

import mindspore.nn as nn
from mindspore import Tensor

from ...register import register
from .loss import Loss


@register("loss", extra_name="joint_mse")
class JointsMSELoss(Loss):
    """Joint Mean square error loss.
    It is the MSE loss of heatmaps with extra weight for different channel.

    Args:
        use_target_weight: Use extra weight in loss calculation. Default: False
        reduction: Type of the reduction to be applied to loss. The optional value
            are "mean", "sum" and "none". Default: "mean"

    Inputs:
        | pred: Predictions, in shape [N, K, H, W]
        | target: Ground truth, in shape [N, K, H, W]
        | target_weight: Loss weight, in shape [N, K]

    Outputs:
        | loss: Loss value
    """

    def __init__(
        self, use_target_weight: bool = False, reduction: Optional[str] = "mean"
    ) -> None:
        super().__init__(reduction=reduction)
        self.use_target_weight = use_target_weight
        self.criterion = nn.MSELoss(reduction="none")

    def construct(
        self, pred: Tensor, target: Tensor, target_weight: Optional[Tensor] = None
    ) -> Tensor:
        loss = self.criterion(pred, target)
        if self.use_target_weight:
            loss = self.get_loss(loss, target_weight[..., None, None])
        else:
            loss = self.get_loss(loss)
        return loss


@register("loss", extra_name="joint_mse_with_mask")
class JointsMSELossWithMask(Loss):
    """Joint Mean square error loss with mask.
    Mask-out position will not contribute to the loss.

    Args:
        reduction: Type of the reduction to be applied to loss. The optional value
            are "mean", "sum" and "none". Default: "mean"

    Inputs:
        | pred: Predictions, in shape [N, K, H, W]
        | target: Ground truth, in shape [N, K, H, W]
        | mask: Ground truth Mask, in shape [N, H, W]

    Outputs:
        | loss: Loss value
    """

    def __init__(self, reduction: Optional[str] = "mean") -> None:
        super().__init__(reduction=reduction)
        self.criterion = nn.MSELoss(reduction="none")

    def construct(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        loss = self.criterion(pred, target)
        loss = self.get_loss(loss, mask[:, None, :, :])
        return loss

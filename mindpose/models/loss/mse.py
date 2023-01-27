from typing import Optional

import mindspore.nn as nn
from mindspore import Tensor

from ...register import register
from .loss import Loss


@register("loss", extra_name="joint_mse")
class JointsMSELoss(Loss):
    """Joint Mean square error loss

    Args:
        use_target_weight: Use extra weight in loss calculation

    Inputs:
        pred: predictions, in shape [N, C, H, W]
        target: ground truth, in shape [N, C, H, W]
        target_weight: loss, weight, in shape [N, C]

    Outputs:
        loss: loss value
    """

    def __init__(self, use_target_weight: bool = False) -> None:
        super().__init__()
        self.criterion = nn.MSELoss(reduction="mean")
        self.use_target_weight = use_target_weight

    def construct(
        self, pred: Tensor, target: Tensor, target_weight: Optional[Tensor] = None
    ) -> Tensor:
        if self.use_target_weight:
            target_weight = target_weight[..., None, None]
            loss = self.criterion(pred * target_weight, target * target_weight)
        else:
            loss = self.criterion(pred, target)
        return loss

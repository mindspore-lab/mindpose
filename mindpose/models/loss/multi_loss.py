from typing import List, Tuple

import mindspore.ops as ops
from mindspore import Tensor

from ...register import register
from .ae import AELoss
from .loss import Loss
from .mse import JointsMSELossWithMask


@register("loss", extra_name="ae_multi_loss")
class AEMultiLoss(Loss):
    """Combined loss of MSE and AE for multi levels of resolutions

    Args:
        num_joints: Number of joints. Default: 17
        num_stages: Number of resolution levels. Default: 2
        stage_sizes: The sizes in each stage. Default: [(128, 128), (256, 256)]
        mse_loss_factor: Weighting for MSE loss at each level. Default: [1.0, 1.0]
        ae_loss_factor: Weighting for Associative embedding loss at each level.
            Default: [0.001, 0.001]
        with_mse_loss: Whether each level involves calculating MSE loss.
            Default: [True, False]
        with_ae_loss: Whether each level involves calculating AE loss.
            Default: [True, False]
        tag_per_joint: Whether each of the joint has its own coordinate encoding.
            Default: True

    Inputs:
        | pred: List of prediction result at each resolution level. In shape [N, aK,
            H, W]. Where K stands for the number of joints. a=2 if the correspoinding
            with_ae_loss is True
        | target: Ground truth of heatmap. In shape [N, S, K, H, W]. Where S stands for
            the number of resolution levels.
        | mask: Ground truth of the heatmap mask. In shape [N, S, H, W].
        | tag_ind: Ground truth of tag position. In shape [N, S, M, K, 2]. Where M
            stands for number of instances.

    Outputs:
        | loss: Single Loss value
    """

    def __init__(
        self,
        num_joints: int = 17,
        num_stages: int = 2,
        stage_sizes: List[Tuple[int, int]] = [(128, 128), (256, 256)],
        mse_loss_factor: List[float] = [1.0, 1.0],
        ae_loss_factor: List[float] = [0.001, 0.001],
        with_mse_loss: List[bool] = [True, True],
        with_ae_loss: List[bool] = [True, False],
        tag_per_joint: bool = True,
    ) -> None:
        super().__init__()
        self.mse_criterion = JointsMSELossWithMask()
        self.ae_criterion = AELoss(tag_per_joint=tag_per_joint)

        self.num_joints = num_joints
        self.num_stages = num_stages
        self.stage_sizes = stage_sizes
        self.mse_loss_factor = mse_loss_factor
        self.ae_loss_factor = ae_loss_factor
        self.with_mse_loss = with_mse_loss
        self.with_ae_loss = with_ae_loss
        self.tag_per_joint = tag_per_joint

    def construct(
        self,
        preds: List[Tensor],
        target: Tensor,
        mask: Tensor,
        tag_ind: Tensor,
    ) -> Tensor:
        total_mse_loss = 0.0
        total_push_loss = 0.0
        total_pull_loss = 0.0

        for i in range(self.num_stages):
            W, H = self.stage_sizes[i]
            pred = preds[i]
            if self.with_mse_loss[i]:
                mse_loss = (
                    self.mse_criterion(
                        pred[:, : self.num_joints],
                        target[:, i, :, :H, :W],
                        mask[:, i, :H, :W],
                    )
                    * self.mse_loss_factor[i]
                )
                total_mse_loss += mse_loss

            if self.with_ae_loss[i]:
                if self.tag_per_joint:
                    push_loss, pull_loss = (
                        self.ae_criterion(pred[:, self.num_joints :], tag_ind[:, i])
                        * self.ae_loss_factor[i]
                    )
                else:
                    push_loss, pull_loss = (
                        self.ae_criterion(pred[:, self.num_joints], tag_ind[:, i])
                        * self.ae_loss_factor[i]
                    )
                total_push_loss += push_loss
                total_pull_loss += pull_loss

        return ops.stack([total_mse_loss, total_push_loss, total_pull_loss])

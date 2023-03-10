from typing import List, Tuple

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

    Inputs:
        | pred: List of prediction result at each resolution level. In shape [N, aK,
            H, W]. Where K stands for the number of joints. a=2 if the correspoinding
            with_ae_loss is True
        | target: Ground truth of heatmap. In shape [N, S, K, H, W]. Where S stands for
            the number of resolution levels.
        | mask: Ground truth of the heatmap mask. In shape [N, S, H, W].
        | tag_mask: Ground truth of tag position. In shape [N, S, M, K, H, W]. Where M
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
    ) -> None:
        super().__init__()
        self.mse_criterion = JointsMSELossWithMask()
        self.ae_criterion = AELoss(reduce=True)

        self.num_joints = num_joints
        self.num_stages = num_stages
        self.stage_sizes = stage_sizes
        self.mse_loss_factor = mse_loss_factor
        self.ae_loss_factor = ae_loss_factor
        self.with_mse_loss = with_mse_loss
        self.with_ae_loss = with_ae_loss

        if any(self.with_ae_loss):
            if not with_ae_loss[0]:
                raise ValueError(
                    "0th element of `with_ae_loss` must be true "
                    "in case when ae_loss is used."
                )

    def construct(
        self,
        pred: List[Tensor],
        target: Tensor,
        mask: Tensor,
        tag_mask: Tensor,
    ) -> Tensor:
        loss = 0.0

        for i in range(self.num_stages):
            W, H = self.stage_sizes[i]
            if self.with_mse_loss[i]:
                loss += (
                    self.mse_criterion(
                        pred[i][:, : self.num_joints, ...],
                        target[:, i, :, :H, :W],
                        mask[:, i, :H, :W],
                    )
                    * self.mse_loss_factor[i]
                )

            if self.with_ae_loss[i]:
                loss += (
                    self.ae_criterion(
                        pred[i][:, self.num_joints :, ...], tag_mask[:, i, ...]
                    )
                    * self.ae_loss_factor[i]
                )
        return loss

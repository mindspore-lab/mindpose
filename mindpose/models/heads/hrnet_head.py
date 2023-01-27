"""
MindSpore implementation of `HRNet` head.
Refer to `Deep High-Resolution Representation Learning for Human Pose Estimation`
"""

import mindspore.nn as nn
from mindspore import Tensor

from ...register import register
from .head import Head


@register("head", extra_name="hrnet_head")
class HRNetHead(Head):
    r"""HRNet Head, based on
    `"Deep High-Resolution Representation Learning for Human Pose Estimation" <https://arxiv.org/abs/1512.03385>`_

    Args:
        in_channels: Number the channels of the input. Default: 32.
        num_joints: Number of joints in the final output. Default: 17
        final_conv_kernel_size: The kernel size in the final convolution layer. Default: 1

    Inputs:
        x: Input Tensor

    Outputs:
        result: Result Tensor
    """

    def __init__(
        self,
        in_channels: int = 32,
        num_joints: int = 17,
        final_conv_kernel_size: int = 1,
    ) -> None:
        super().__init__()
        self.head = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_joints,
            kernel_size=final_conv_kernel_size,
            has_bias=True,
        )

    def construct(self, x: Tensor) -> Tensor:
        x = self.head(x)
        return x

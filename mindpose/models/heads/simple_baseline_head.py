"""
MindSpore implementation of `SimpleBaseline` head.
Refer to `Simple Baselines for Human Pose Estimation and Tracking`
"""


from typing import List

import mindspore.nn as nn
from mindspore import Tensor

from ...register import register
from .head import Head


@register("head", extra_name="simple_baseline_head")
class SimpleBaselineHead(Head):
    r"""SimpleBaseline Head, based on
    `"Simple Baselines for Human Pose Estimation and Tracking"
    <https://arxiv.org/abs/1804.06208>`_.
    It contains few number of deconvolution layers following by a 1x1 convolution layer.

    Args:
        num_deconv_layers: Number of deconvolution layers. Default: 3
        num_deconv_filters: Number of filters in each deconvolution layer.
            Default: [256, 256, 256]
        num_deconv_kernels: Kernel size in each deconvolution layer. Default: [4, 4, 4]
        in_channels: number the channels of the input. Default: 2048.
        num_joints: Number of joints in the final output. Default: 17
        final_conv_kernel_size: The kernel size in the final convolution layer.
            Default: 1

    Inputs:
        | x: Input Tensor

    Outputs:
        | result: Result Tensor
    """

    def __init__(
        self,
        num_deconv_layers: int = 3,
        num_deconv_filters: List[int] = [256, 256, 256],
        num_deconv_kernels: List[int] = [4, 4, 4],
        in_channels: int = 2048,
        num_joints: int = 17,
        final_conv_kernel_size: int = 1,
    ) -> None:
        super().__init__()
        self.num_deconv_layers = num_deconv_layers
        self.num_deconv_filters = num_deconv_filters
        self.num_deconv_kernels = num_deconv_kernels
        self.in_channels = in_channels

        self.deconv_layer = self.make_deconv_layer()

        self.final_layer = nn.Conv2d(
            in_channels=self.num_deconv_filters[-1],
            out_channels=num_joints,
            kernel_size=final_conv_kernel_size,
            has_bias=True,
        )

    def _get_deconv_padding(self, deconv_kernel):
        if deconv_kernel == 4:
            padding = 1
        elif deconv_kernel == 2:
            padding = 0
        return padding

    def make_deconv_layer(self):
        layers = []
        in_channels = self.in_channels
        for i in range(self.num_deconv_layers):
            padding = self._get_deconv_padding(self.num_deconv_kernels[i])
            planes = self.num_deconv_filters[i]
            layers.append(
                nn.Conv2dTranspose(
                    in_channels=in_channels,
                    out_channels=planes,
                    kernel_size=self.num_deconv_kernels[i],
                    stride=2,
                    pad_mode="pad",
                    padding=padding,
                )
            )
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU())
            in_channels = planes

        return nn.SequentialCell(*layers)

    def construct(self, x: Tensor) -> Tensor:
        x = self.deconv_layer(x)
        x = self.final_layer(x)
        return x

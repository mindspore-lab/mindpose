"""
MindSpore implementation of `HigherHRNet` head.
Refer to
`HigherHRNet: Scale-Aware Representation Learning for Bottom-Up Human Pose Estimation`
"""
from typing import List, Optional

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

from ...register import register
from .head import Head


class BasicBlock(nn.Cell):
    """Basic block of HRNet"""

    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        channels: int,
        stride: int = 1,
        groups: int = 1,
        base_width: int = 64,
        norm: Optional[nn.Cell] = None,
        down_sample: Optional[nn.Cell] = None,
    ) -> None:
        super().__init__()
        if norm is None:
            norm = nn.BatchNorm2d
        assert groups == 1, "BasicBlock only supports groups=1"
        assert base_width == 64, "BasicBlock only supports base_width=64"

        self.conv1 = nn.Conv2d(
            in_channels,
            channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            pad_mode="pad",
        )
        self.bn1 = norm(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1, pad_mode="pad"
        )
        self.bn2 = norm(channels)
        self.down_sample = down_sample

    def construct(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.down_sample is not None:
            identity = self.down_sample(x)

        out += identity
        out = self.relu(out)

        return out


@register("head", extra_name="higher_hrnet_head")
class HigherHRNetHead(Head):
    r"""HigherHRNet Head, based on
    `"HigherHRNet: Scale-Aware Representation Learning for Bottom-Up
    Human Pose Estimation" <https://arxiv.org/abs/1908.10357>`_.

    Args:
        in_channels: Number the channels of the input. Default: 32.
        num_joints: Number of joints in the final output. Default: 17
        with_ae_loss: Output the associated embedding for each resolution.
            Default: [True, False]
        tag_per_joint: Wether each of the joint has its own coordinate encoding.
            Default: True
        final_conv_kernel_size: The kernel size in the final convolution layer.
            Default: 1
        num_deconv_layers: Number of deconvolution layers. Default: 1
        num_deconv_filters: Number of filters in each deconvolution layer.
            Default: [32]
        num_deconv_kernels: Kernel size in each deconvolution layer. Default: [4]
        cat_outputs: Whether to concate the feature before deconvolution layer at
            each resoluton. Default: [True]
        num_basic_blocks: Number of basic blocks after deconvolution. Default: 4

    Inputs:
        | x: Input Tensor

    Outputs:
        | result: Tuples of Tensor at different resolution
    """

    def __init__(
        self,
        in_channels: int = 32,
        num_joints: int = 17,
        with_ae_loss: List[bool] = [True, False],
        tag_per_joint: bool = True,
        final_conv_kernel_size: int = 1,
        num_deconv_layers: int = 1,
        num_deconv_filters: List[int] = [32],
        num_deconv_kernels: List[int] = [4],
        cat_outputs: List[bool] = [True],
        num_basic_blocks: int = 4,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_joints = num_joints
        self.with_ae_loss = with_ae_loss
        self.tag_per_joint = tag_per_joint
        self.final_conv_kernel_size = final_conv_kernel_size
        self.num_deconv_layers = num_deconv_layers
        self.num_deconv_filters = num_deconv_filters
        self.num_deconv_kernels = num_deconv_kernels
        self.cat_outputs = cat_outputs
        self.num_basic_blocks = num_basic_blocks

        self.deconv_layers = self._make_deconv_layers()
        self.final_layers = self._make_final_layers()

    def _get_deconv_padding(self, deconv_kernel: int) -> int:
        if deconv_kernel == 4:
            padding = 1
        elif deconv_kernel == 2:
            padding = 0
        return padding

    def _make_deconv_layers(self) -> nn.CellList:
        dim_tag = self.num_joints if self.tag_per_joint else 1
        input_channels = self.in_channels

        deconv_layers = []
        for i in range(self.num_deconv_layers):
            if self.cat_outputs[i]:
                if self.with_ae_loss[i]:
                    final_output_channels = self.num_joints + dim_tag
                else:
                    final_output_channels = self.num_joints

                input_channels += final_output_channels
            output_channels = self.num_deconv_filters[i]
            padding = self._get_deconv_padding(self.num_deconv_kernels[i])

            layers = []
            layers.append(
                nn.SequentialCell(
                    nn.Conv2dTranspose(
                        in_channels=input_channels,
                        out_channels=output_channels,
                        kernel_size=self.num_deconv_kernels[i],
                        stride=2,
                        padding=padding,
                        pad_mode="pad",
                    ),
                    nn.BatchNorm2d(output_channels),
                    nn.ReLU(),
                )
            )
            for _ in range(self.num_basic_blocks):
                layers.append(BasicBlock(output_channels, output_channels))
            deconv_layers.append(nn.SequentialCell(*layers))
            input_channels = output_channels

        return nn.CellList(deconv_layers)

    def _make_final_layers(self) -> nn.CellList:
        dim_tag = self.num_joints if self.tag_per_joint else 1
        if self.with_ae_loss[0]:
            output_channels = self.num_joints + dim_tag
        else:
            output_channels = self.num_joints

        final_layers = []
        final_layers.append(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=output_channels,
                kernel_size=self.final_conv_kernel_size,
                padding=1 if self.final_conv_kernel_size == 3 else 0,
                pad_mode="pad",
                has_bias=True,
            )
        )

        for i in range(self.num_deconv_layers):
            input_channels = self.num_deconv_filters[i]

            if self.with_ae_loss[i + 1]:
                output_channels = self.num_joints + dim_tag
            else:
                output_channels = self.num_joints

            final_layers.append(
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=self.final_conv_kernel_size,
                    padding=1 if self.final_conv_kernel_size == 3 else 0,
                    pad_mode="pad",
                    has_bias=True,
                )
            )

        return nn.CellList(final_layers)

    def construct(self, x: Tensor) -> List[Tensor]:
        final_outputs = []
        y = self.final_layers[0](x)
        final_outputs.append(y)

        for i in range(self.num_deconv_layers):
            if self.cat_outputs[i]:
                x = ops.concat((x, y), axis=1)

            x = self.deconv_layers[i](x)
            y = self.final_layers[i + 1](x)
            final_outputs.append(y)

        return final_outputs

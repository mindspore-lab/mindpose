# TODO: resnet need to import from mindcv

from typing import List, Optional, Type, Union

import mindspore.nn as nn
from mindspore import Tensor

from ...register import register

from .backbone import Backbone

__all__ = ["ResNet", "resnet50", "resnet101", "resnet152"]


class BasicBlock(nn.Cell):
    """define the basic block of resnet"""

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


class Bottleneck(nn.Cell):
    """
    Bottleneck here places the stride for downsampling at 3x3convolution(self.conv2)
    as torchvision does, while original implementation places the stride at the first
    1x1 convolution(self.conv1)
    """

    expansion: int = 4

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

        width = int(channels * (base_width / 64.0)) * groups

        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, stride=1)
        self.bn1 = norm(width)
        self.conv2 = nn.Conv2d(
            width,
            width,
            kernel_size=3,
            stride=stride,
            padding=1,
            pad_mode="pad",
            group=groups,
        )
        self.bn2 = norm(width)
        self.conv3 = nn.Conv2d(
            width, channels * self.expansion, kernel_size=1, stride=1
        )
        self.bn3 = norm(channels * self.expansion)
        self.relu = nn.ReLU()
        self.down_sample = down_sample

    def construct(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.down_sample is not None:
            identity = self.down_sample(x)

        out += identity
        out = self.relu(out)

        return out


@register("backbone")
class ResNet(Backbone):
    r"""ResNet model class, based on
    `"Deep Residual Learning for Image Recognition"
    <https://arxiv.org/abs/1512.03385>`_.

    Args:
        block: Block of resnet
        layers: Number of layers of each stage
        in_channels: Number the channels of the input. Default: 3
        groups: Number of groups for group conv in blocks. Default: 1
        base_width: Base width of pre group hidden channel in blocks. Default: 64
        norm: Normalization layer in blocks. Default: None

    Inputs:
        | x: Input Tensor

    Outputs:
        | feature: Feature Tensor
    """

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        in_channels: int = 3,
        groups: int = 1,
        base_width: int = 64,
        norm: Optional[nn.Cell] = None,
    ) -> None:
        super().__init__()
        if norm is None:
            norm = nn.BatchNorm2d

        self.norm: nn.Cell = norm  # add type hints to make pylint happy
        self.input_channels = 64
        self.groups = groups
        self.base_with = base_width

        self.conv1 = nn.Conv2d(
            in_channels,
            self.input_channels,
            kernel_size=7,
            stride=2,
            pad_mode="pad",
            padding=3,
        )
        self.bn1 = norm(self.input_channels)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        channels: int,
        block_nums: int,
        stride: int = 1,
    ) -> nn.SequentialCell:
        """build model depending on cfgs"""
        down_sample = None

        if stride != 1 or self.input_channels != channels * block.expansion:
            down_sample = nn.SequentialCell(
                [
                    nn.Conv2d(
                        self.input_channels,
                        channels * block.expansion,
                        kernel_size=1,
                        stride=stride,
                    ),
                    self.norm(channels * block.expansion),
                ]
            )

        layers = []
        layers.append(
            block(
                self.input_channels,
                channels,
                stride=stride,
                down_sample=down_sample,
                groups=self.groups,
                base_width=self.base_with,
                norm=self.norm,
            )
        )
        self.input_channels = channels * block.expansion

        for _ in range(1, block_nums):
            layers.append(
                block(
                    self.input_channels,
                    channels,
                    groups=self.groups,
                    base_width=self.base_with,
                    norm=self.norm,
                )
            )

        return nn.SequentialCell(layers)

    def forward_feature(self, x: Tensor) -> Tensor:
        """Perform the feature extraction.

        Args:
            x: Tensor

        Returns:
            Extracted feature
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    @property
    def out_channels(self) -> int:
        """Get number of output channels.

        Returns:
            Output channels.
        """
        return 512 * Bottleneck.expansion


@register("backbone")
def resnet50(
    pretrained: bool = False, ckpt_url: str = "", in_channels: int = 3, **kwargs
) -> ResNet:
    """Get 50 layers ResNet model.

    Args:
        pretrained: Whether the model is pretrained. Default: False
        ckpt_url: Url of the pretrained weight. Default: ""
        in_channels: Number of input channels. Default: 3
        kwargs: Arguments which feed into Resnet class

    Returns:
        Resnet model
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], in_channels=in_channels, **kwargs)

    return model


@register("backbone")
def resnet101(
    pretrained: bool = False, ckpt_url: str = "", in_channels: int = 3, **kwargs
) -> ResNet:
    """Get 101 layers ResNet model.

    Args:
        pretrained: Whether the model is pretrained. Default: False
        ckpt_url: Url of the pretrained weight. Default: ""
        in_channels: Number of input channels. Default: 3
        kwargs: Arguments which feed into Resnet class

    Returns:
        Resnet model
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], in_channels=in_channels, **kwargs)

    return model


@register("backbone")
def resnet152(
    pretrained: bool = False, ckpt_url: str = "", in_channels: int = 3, **kwargs
) -> ResNet:
    """Get 152 layers ResNet model.

    Args:
        pretrained: Whether the model is pretrained. Default: False
        ckpt_url: Url of the pretrained weight. Default: ""
        in_channels: Number of input channels. Default: 3
        kwargs: Arguments which feed into Resnet class

    Returns:
        Resnet model
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], in_channels=in_channels, **kwargs)

    return model

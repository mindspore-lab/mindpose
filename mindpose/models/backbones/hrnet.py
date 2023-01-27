"""
MindSpore implementation of `HRNet` backbone.
Refer to Deep High-Resolution Representation Learning for Human Pose Estimation
"""
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

from ...register import register

from .backbone import Backbone


__all__ = ["HRNet", "hrnet_w32", "hrnet_w48"]


class IdentityCell(nn.Cell):
    """Identity Cell"""

    def __init__(self) -> None:
        super().__init__()

    def construct(self, x: Any) -> Any:
        return x


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


class Bottleneck(nn.Cell):
    """Bottleneck block of HRNet"""

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


class HRModule(nn.Cell):
    """High-Resolution Module for HRNet.
    In this module, every branch has 4 BasicBlocks/Bottlenecks. Fusion/Exchange
    is in this module.
    """

    def __init__(
        self,
        num_branches: int,
        block: Type[Union[BasicBlock, Bottleneck]],
        num_blocks: List[int],
        num_inchannels: List[int],
        num_channels: List[int],
        multi_scale_output: bool = True,
    ) -> None:
        super().__init__()
        self._check_branches(num_branches, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, block, num_blocks, num_channels
        )
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU()

    @staticmethod
    def _check_branches(
        num_branches: int,
        num_blocks: List[int],
        num_inchannels: List[int],
        num_channels: List[int],
    ) -> None:
        """Check input to avoid ValueError."""
        if num_branches != len(num_blocks):
            error_msg = f"NUM_BRANCHES({num_branches})!= NUM_BLOCKS({len(num_blocks)})"
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = (
                f"NUM_BRANCHES({num_branches})!= NUM_CHANNELS({len(num_channels)})"
            )
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = (
                f"NUM_BRANCHES({num_branches}) != NUM_INCHANNELS({len(num_inchannels)})"
            )
            raise ValueError(error_msg)

    def _make_one_branch(
        self,
        branch_index: int,
        block: Type[Union[BasicBlock, Bottleneck]],
        num_blocks: List[int],
        num_channels: List[int],
        stride: int = 1,
    ) -> nn.SequentialCell:
        downsample = None
        if (
            stride != 1
            or self.num_inchannels[branch_index]
            != num_channels[branch_index] * block.expansion
        ):
            downsample = nn.SequentialCell(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1,
                    stride=stride,
                ),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                down_sample=downsample,
            )
        )
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for _ in range(1, num_blocks[branch_index]):
            layers.append(
                block(self.num_inchannels[branch_index], num_channels[branch_index])
            )

        return nn.SequentialCell(layers)

    def _make_branches(
        self,
        num_branches: int,
        block: Type[Union[BasicBlock, Bottleneck]],
        num_blocks: List[int],
        num_channels: List[int],
    ) -> nn.CellList:
        """Make branches."""
        branches = []

        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.CellList(branches)

    def _make_fuse_layers(self) -> nn.CellList:
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.SequentialCell(
                            nn.Conv2d(
                                num_inchannels[j], num_inchannels[i], kernel_size=1
                            ),
                            nn.BatchNorm2d(num_inchannels[i]),
                        )
                    )
                elif j == i:
                    fuse_layer.append(IdentityCell())
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.SequentialCell(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        pad_mode="pad",
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3),
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.SequentialCell(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        pad_mode="pad",
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3),
                                    nn.ReLU(),
                                )
                            )
                    fuse_layer.append(nn.SequentialCell(conv3x3s))
            fuse_layers.append(nn.CellList(fuse_layer))

        return nn.CellList(fuse_layers)

    def construct(self, x: List[Tensor]) -> List[Tensor]:
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    _, _, height, width = x[i].shape
                    t = self.fuse_layers[i][j](x[j])
                    t = ops.ResizeNearestNeighbor((height, width))(t)
                    y = y + t
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        if not self.multi_scale_output:
            x_fuse = x_fuse[0]

        return x_fuse


@register("backbone")
class HRNet(Backbone):
    r"""HRNet Backbone, based on
    `"Deep High-Resolution Representation Learning for Human Pose Estimation" <https://arxiv.org/abs/1512.03385>`_

    Args:
        stage_cfg: Configuration of the extra blocks
        in_channels: Number the channels of the input. Default: 3.

    Inputs:
        x: Input Tensor

    Outputs:
        feature: Feature Tensor
    """

    blocks_dict = {"BASIC": BasicBlock, "BOTTLENECK": Bottleneck}

    def __init__(
        self, stage_cfg: Dict[str, Dict[str, int]], in_channels: int = 3
    ) -> None:
        super().__init__()

        self.stage_cfg = stage_cfg
        # stem net
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=2, padding=1, pad_mode="pad"
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(
            64, 64, kernel_size=3, stride=2, padding=1, pad_mode="pad"
        )
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        # stage 1
        self.stage1_cfg = self.stage_cfg["stage1"]
        num_channels = self.stage1_cfg["num_channels"][0]
        num_blocks = self.stage1_cfg["num_blocks"][0]
        block = self.blocks_dict[self.stage1_cfg["block"]]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)

        # stage 2
        self.stage2_cfg = self.stage_cfg["stage2"]
        num_channels = self.stage2_cfg["num_channels"]
        block = self.blocks_dict[self.stage2_cfg["block"]]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]

        self.transition1, self.transition1_flags = self._make_transition_layer(
            [256], num_channels
        )
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels
        )

        # stage 3
        self.stage3_cfg = self.stage_cfg["stage3"]
        num_channels = self.stage3_cfg["num_channels"]
        block = self.blocks_dict[self.stage3_cfg["block"]]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]

        self.transition2, self.transition2_flags = self._make_transition_layer(
            pre_stage_channels, num_channels
        )
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels
        )

        # stage 4
        self.stage4_cfg = self.stage_cfg["stage4"]
        num_channels = self.stage4_cfg["num_channels"]
        block = self.blocks_dict[self.stage4_cfg["block"]]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3, self.transition3_flags = self._make_transition_layer(
            pre_stage_channels, num_channels
        )
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg,
            num_channels,
            multi_scale_output=self.stage4_cfg.get("multiscale_output", False),
        )

    def _make_transition_layer(
        self, num_channels_pre_layer: List[int], num_channels_cur_layer: List[int]
    ) -> Tuple[nn.CellList, List[bool]]:
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        transition_layers_flags = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.SequentialCell(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                kernel_size=3,
                                padding=1,
                                pad_mode="pad",
                            ),
                            nn.BatchNorm2d(num_channels_cur_layer[i]),
                            nn.ReLU(),
                        )
                    )
                    transition_layers_flags.append(True)
                else:
                    transition_layers.append(IdentityCell())
                    transition_layers_flags.append(False)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = (
                        num_channels_cur_layer[i]
                        if j == i - num_branches_pre
                        else inchannels
                    )
                    conv3x3s.append(
                        nn.SequentialCell(
                            [
                                nn.Conv2d(
                                    inchannels,
                                    outchannels,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    pad_mode="pad",
                                ),
                                nn.BatchNorm2d(outchannels),
                                nn.ReLU(),
                            ]
                        )
                    )
                transition_layers.append(nn.SequentialCell(conv3x3s))
                transition_layers_flags.append(True)

        return nn.CellList(transition_layers), transition_layers_flags

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        in_channels: int,
        out_channels: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.SequentialCell:
        downsample = None
        if stride != 1 or in_channels != out_channels * block.expansion:
            downsample = nn.SequentialCell(
                nn.Conv2d(
                    in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(in_channels, out_channels, stride, down_sample=downsample))
        for _ in range(1, blocks):
            layers.append(block(out_channels * block.expansion, out_channels))

        return nn.SequentialCell(layers)

    def _make_stage(
        self,
        layer_config: Dict[str, int],
        num_inchannels: int,
        multi_scale_output: bool = True,
    ) -> Tuple[nn.SequentialCell, int]:
        num_modules = layer_config["num_modules"]
        num_branches = layer_config["num_branches"]
        num_blocks = layer_config["num_blocks"]
        num_channels = layer_config["num_channels"]
        block = self.blocks_dict[layer_config["block"]]

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HRModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    reset_multi_scale_output,
                )
            )
            num_inchannels = modules[-1].num_inchannels

        return nn.SequentialCell(modules), num_inchannels

    @property
    def out_channels(self) -> int:
        return self.stage4_cfg["num_channels"][0]

    def construct(self, x: Tensor) -> List[Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # stage 1
        x = self.layer1(x)

        # stage 2
        x_list = []
        for i in range(self.stage2_cfg["num_branches"]):
            if self.transition1_flags[i]:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        # stage 3
        x_list = []
        for i in range(self.stage3_cfg["num_branches"]):
            if self.transition2_flags[i]:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        # stage 4
        x_list = []
        for i in range(self.stage4_cfg["num_branches"]):
            if self.transition3_flags[i]:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        return y_list


@register("backbone")
def hrnet_w32(
    pretrained: bool = False, ckpt_url: str = "", in_channels: int = 3
) -> HRNet:
    """Get HRNet with width=32 model."""
    stage_cfg = dict(
        stage1=dict(
            num_modules=1,
            num_branches=1,
            block="BOTTLENECK",
            num_blocks=[4],
            num_channels=[64],
        ),
        stage2=dict(
            num_modules=1,
            num_branches=2,
            block="BASIC",
            num_blocks=[4, 4],
            num_channels=[32, 64],
        ),
        stage3=dict(
            num_modules=4,
            num_branches=3,
            block="BASIC",
            num_blocks=[4, 4, 4],
            num_channels=[32, 64, 128],
        ),
        stage4=dict(
            num_modules=3,
            num_branches=4,
            block="BASIC",
            num_blocks=[4, 4, 4, 4],
            num_channels=[32, 64, 128, 256],
            multiscale_output=False,
        ),
    )
    model = HRNet(stage_cfg, in_channels=in_channels)

    # TODO: add pretrain
    return model


@register("backbone")
def hrnet_w48(
    pretrained: bool = False, ckpt_url: str = "", in_channels: int = 3
) -> HRNet:
    """Get HRNet with width=48 model."""
    stage_cfg = dict(
        stage1=dict(
            num_modules=1,
            num_branches=1,
            block="BOTTLENECK",
            num_blocks=[4],
            num_channels=[64],
        ),
        stage2=dict(
            num_modules=1,
            num_branches=2,
            block="BASIC",
            num_blocks=[4, 4],
            num_channels=[48, 96],
        ),
        stage3=dict(
            num_modules=4,
            num_branches=3,
            block="BASIC",
            num_blocks=[4, 4, 4],
            num_channels=[48, 96, 192],
        ),
        stage4=dict(
            num_modules=3,
            num_branches=4,
            block="BASIC",
            num_blocks=[4, 4, 4, 4],
            num_channels=[48, 96, 192, 384],
            multiscale_output=False,
        ),
    )
    model = HRNet(stage_cfg, in_channels=in_channels)
    return model

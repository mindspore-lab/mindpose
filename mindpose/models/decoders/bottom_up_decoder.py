from typing import List, Tuple, Union

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

from mindspore import Tensor

from ...register import register
from .decoder import Decoder


@register("decoder", extra_name="bottomup_heatmap_ae")
class BottomUpHeatMapAEDecoder(Decoder):
    """Decode the heatmaps with associativa embedding into coordinates

    Args:
        num_joints: Number of joints. Default: 17
        num_stages: Number of resolution in the heatmap outputs. If it is larger than
            one, then heatmap aggregation is performed. Default: 2
        with_ae_loss: Output the associated embedding for each resolution.
            Default: [True, False]
        use_nms: Apply NMS for the heatmap output. Default: False
        nms_kernel: NMS kerrnel size. Default: 5
        max_num: Maximum number (K) of instances in the image. Default: 30
        tag_per_joint: Whether each of the joint has its own coordinate encoding.
            Default: True

    Inputs:
        | model_output: Model output. It is a list of Tensors with the length equal to
            the num_stages.
        | mask: Heatmap mask of the valid region.

    Outputs:
        | val_k, tag_k, ind_k: Tuples contains the maximum value of the heatmap for each
            joint with the corresponding tag value and location.
    """

    def __init__(
        self,
        num_joints: int = 17,
        num_stages: int = 2,
        with_ae_loss: List[bool] = [True, False],
        use_nms: bool = False,
        nms_kernel: int = 5,
        max_num: int = 30,
        tag_per_joint: bool = True,
    ) -> None:
        super().__init__()
        self.num_joints = num_joints
        self.num_stages = num_stages
        self.with_ae_loss = with_ae_loss
        self.use_nms = use_nms
        self.nms_kernel = nms_kernel
        self.max_num = max_num
        self.tag_per_joint = tag_per_joint

        if self.use_nms:
            self.pool = nn.MaxPool2d(kernel_size=self.nms_kernel, pad_mode="same")
        else:
            self.pool = None

    def construct(
        self, model_output: List[Tensor], mask: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        heatmap, tagging_heatmap = list(), list()
        for i in range(self.num_stages):
            heatmap.append(model_output[i][:, : self.num_joints])
            if self.with_ae_loss[i]:
                tagging_heatmap.append(model_output[i][:, self.num_joints :])
        # use the tagging heatmap with the lowest resolution only.
        return self._decode(heatmap, tagging_heatmap[0], mask)

    def _decode(
        self,
        heatmap: Union[Tensor, List[Tensor]],
        tagging_heatmap: Tensor,
        mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        mask = mask[:, None, ...]

        if self.num_stages > 1:
            heatmap = self._aggregate_heatmap(heatmap)

        if self.use_nms:
            heatmap = self._nms(heatmap)

        # resize the tagging_heatmap to the same resolution as heatmap
        _, _, H, W = heatmap.shape
        tagging_heatmap = ops.ResizeBilinear((H, W))(tagging_heatmap)
        mask = ops.ResizeNearestNeighbor((H, W))(ops.cast(mask, tagging_heatmap.dtype))

        # mask out nonvalid heatmap region
        mask = ops.cast(mask, ms.bool_)
        heatmap = ops.masked_fill(heatmap, ~mask, 0)

        val_k, tag_k, ind_k = self._get_max_preds(heatmap, tagging_heatmap)
        return val_k, tag_k, ind_k

    def _aggregate_heatmap(self, heatmaps: List[Tensor]) -> Tensor:
        """Aggretate the heatmaps with multi-resolutions"""
        # assume the last heatmap has the largest resolution
        base_heatmap = heatmaps[-1]
        _, _, H, W = base_heatmap.shape
        for i in range(self.num_stages - 1):
            heatmap = ops.ResizeBilinear((H, W))(heatmaps[i])
            base_heatmap += heatmap
        base_heatmap /= self.num_stages
        return base_heatmap

    def _get_max_preds(
        self, heatmap: Tensor, tagging_heatmap: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Get the maximum value of the heatmap for each joint
        with the corresponding tag value and location"""
        N, K, H, W = heatmap.shape
        heatmap = heatmap.reshape(N, K, -1)
        val_k, ind = ops.top_k(heatmap, k=self.max_num)

        K_1 = tagging_heatmap.shape[1]
        tagging_heatmap = tagging_heatmap.reshape(N, K_1, W * H, -1)
        if not self.tag_per_joint:
            tagging_heatmap = ops.broadcast_to(tagging_heatmap, (-1, K, -1, -1))

        tag_k = list()
        for i in range(tagging_heatmap.shape[3]):
            tag_k.append(ops.gather_elements(tagging_heatmap[..., i], 2, ind))
        tag_k = ops.stack(tag_k, axis=3)

        x = ind % W
        y = ind // W

        ind_k = ops.stack((x, y), axis=3)
        ind_k = ops.cast(ind_k, val_k.dtype)
        return val_k, tag_k, ind_k

    def _nms(self, heatmap: Tensor) -> Tensor:
        """Perform nms on heatmap"""
        heatmap_max = self.pool(heatmap)
        heatmap_ind = ops.equal(heatmap_max, heatmap)
        heatmap = heatmap * ops.cast(heatmap_ind, heatmap.dtype)
        return heatmap

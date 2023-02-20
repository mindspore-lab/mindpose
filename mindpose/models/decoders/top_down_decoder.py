from typing import Tuple

import mindspore as ms
import mindspore.ops as ops
from mindspore import Tensor

from ...register import register
from .decoder import Decoder


@register("decoder", extra_name="topdown_heatmap")
class TopDownHeatMapDecoder(Decoder):
    """Decode the heatmaps into coordinates with bounding boxes.

    Args:
        pixel_std: The scaling factor using in decoding. Default: 200.
        to_original: Convert the coordinate into the raw image. Default: True
        shift_coordinate: Perform a +-0.25 pixel coordinate shift based on heatmap
            value. Default: True
        use_udp: Use Unbiased Data Processing (UDP) decoding. Default: False

    Inputs:
        | heatmap: The ordinary output based on heatmap-based model,
            in shape [N, C, H, W]
        | center: Center of the bounding box (x, y) in raw image, in shape [N, C, 2]
        | scale: Scale of the bounding box with respect to the raw image,
            in shape [N, C, 2]
        | score: Score of the bounding box, in shape [N, C, 1]

    Outputs:
        | coordinate: The coordindate of C joints,
            in shape [N, C, 3(x_coord, y_coord, score)]
        | boxes: The coor bounding boxes, in shape
            [N, 6(center_x, center_y, scale_x, scale_y, area, bounding_box_score)]
    """

    def __init__(
        self,
        pixel_std: float = 200.0,
        to_original: bool = True,
        shift_coordinate: bool = True,
        use_udp: bool = False,
    ) -> None:
        super().__init__()
        self.pixel_std = pixel_std
        self.to_original = to_original
        self.shift_coordinate = shift_coordinate
        self.use_udp = use_udp

    def construct(
        self, heatmap: Tensor, center: Tensor, scale: Tensor, score: Tensor
    ) -> Tuple[Tensor, Tensor]:
        batch_size = heatmap.shape[0]

        coords, maxvals, maxvals_mask = self._get_max_preds(heatmap)
        if self.shift_coordinate:
            coords = self._shift_coordinate(coords, heatmap, maxvals_mask)
        if self.to_original:
            coords = self._transform_preds(coords, center, scale, heatmap.shape[2:])

        all_preds = ops.zeros((batch_size, coords.shape[1], 3), ms.float32)
        all_boxes = ops.zeros((batch_size, 6), ms.float32)
        all_preds[:, :, 0:2] = coords[:, :, 0:2]
        all_preds[:, :, 2:3] = maxvals
        all_boxes[:, 0:2] = center[:, 0:2]
        all_boxes[:, 2:4] = scale[:, 0:2]
        all_boxes[:, 4] = ops.prod(scale * self.pixel_std, axis=1)
        all_boxes[:, 5] = score

        return all_preds, all_boxes

    def _get_max_preds(self, heatmap: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Get keypoint predictions from score maps."""
        batch_size = heatmap.shape[0]
        num_joints = heatmap.shape[1]
        width = heatmap.shape[3]
        heatmap = ops.reshape(heatmap, (batch_size, num_joints, -1))
        idx, maxvals = ops.max(heatmap, axis=2, keep_dims=True)

        # a bolean mask storing the location of the maximum value
        maskvals_mask = ops.zeros(heatmap.shape, ms.bool_)
        maskvals_mask = ops.tensor_scatter_elements(
            maskvals_mask, idx, ops.ones(idx.shape, ms.bool_), axis=2
        )
        maskvals_mask = ops.reshape(maskvals_mask, (batch_size, num_joints, -1, width))

        preds = ops.cast(ops.tile(idx, (1, 1, 2)), ms.float32)

        preds[:, :, 0] = preds[:, :, 0] % width
        preds[:, :, 1] = ops.floor((preds[:, :, 1]) / width)

        return preds, maxvals, maskvals_mask

    def _shift_coordinate(
        self, coords: Tensor, heatmap: Tensor, maxvals_mask: Tensor
    ) -> Tensor:
        """shift the coordinate by +- 0.25 pixel towards
        to the location of maximum value"""
        batch_size = coords.shape[0]
        num_joints = coords.shape[1]

        heatmap_diff_x = ops.zeros_like(heatmap)
        heatmap_diff_y = ops.zeros_like(heatmap)
        heatmap_diff_x[:, :, :, 1:-1] = heatmap[:, :, :, 2:] - heatmap[:, :, :, :-2]
        heatmap_diff_y[:, :, 1:-1, :] = heatmap[:, :, 2:, :] - heatmap[:, :, :-2, :]
        heatmap_sign_x = ms.numpy.sign(heatmap_diff_x)
        heatmap_sign_y = ms.numpy.sign(heatmap_diff_y)

        offset_x = ops.masked_select(heatmap_sign_x, maxvals_mask)
        offset_y = ops.masked_select(heatmap_sign_y, maxvals_mask)
        offset_x = ops.reshape(offset_x, (batch_size, num_joints)) * 0.25
        offset_y = ops.reshape(offset_y, (batch_size, num_joints)) * 0.25

        coords[..., 0] += offset_x
        coords[..., 1] += offset_y

        return coords

    def _transform_preds(
        self,
        coords: Tensor,
        center: Tensor,
        scale: Tensor,
        heatmap_shape: Tuple[int, int],
    ) -> Tensor:
        """Get final keypoint predictions from heatmaps and apply scaling and
        translation to map them back to the image."""
        scale = scale * self.pixel_std

        if self.use_udp:
            scale_x = scale[:, 0:1] / (heatmap_shape[1] - 1.0)
            scale_y = scale[:, 1:2] / (heatmap_shape[0] - 1.0)
        else:
            scale_x = scale[:, 0:1] / heatmap_shape[1]
            scale_y = scale[:, 1:2] / heatmap_shape[0]

        target_coords = ops.ones_like(coords)
        target_coords[:, :, 0] = (
            coords[:, :, 0] * scale_x + center[:, 0:1] - scale[:, 0:1] * 0.5
        )
        target_coords[:, :, 1] = (
            coords[:, :, 1] * scale_y + center[:, 1:2] - scale[:, 1:2] * 0.5
        )

        return target_coords

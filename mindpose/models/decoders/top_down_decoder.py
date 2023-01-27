from typing import Tuple

import mindspore as ms
import mindspore.ops as ops
from mindspore import Tensor

from ...register import register
from .decoder import Decoder


@register("decoder", extra_name="topdown_heatmap")
class TopDownHeatMapDecoder(Decoder):
    """Decode the heatmap into coordinate with bounding box

    Args:
        pixel_std: The scaling factor using in decoding. Default: 200.
        to_original: Convert the coordinate to the raw image. Default: True
        shift_coordinate: Perform a +-0.25 coordinate shift based on heatmap value. Default: True

    Inputs:
        heatmap: The ordinary output based on heatmap-based model, in shape [N, H, W]
        center: Center of the bounding box (x, y) in raw image, in shape [N, 2]
        scale: Scale of the bounding box with respect to the raw image, in shape [N, 2]
        score: Score of the bounding box, in shape [N, 1]

    Outputs:
        coordinate: The coordindate of M joints, in shape [N, M, (x_coord, y_coord, score)]
        boxes: The coor bounding boxes, in shape [N, (center_x, center_y, scale_x, scale_y, area, bounding box score)]
    """

    def __init__(
        self,
        pixel_std: float = 200.0,
        to_original: bool = True,
        shift_coordinate: bool = True,
    ) -> None:
        super().__init__()
        self.pixel_std = pixel_std
        self.to_original = to_original
        self.shift_coordinate = shift_coordinate

    def construct(
        self, heatmap: Tensor, center: Tensor, scale: Tensor, score: Tensor
    ) -> Tuple[Tensor, Tensor]:
        batch_size = heatmap.shape[0]

        coords, maxvals = self._get_max_preds(heatmap)
        if self.shift_coordinate:
            coords = self._shift_coordinate(coords, heatmap)
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

    def _get_max_preds(self, heatmap: Tensor) -> Tensor:
        """Get keypoint predictions from score maps."""
        batch_size = heatmap.shape[0]
        num_joints = heatmap.shape[1]
        width = heatmap.shape[3]
        heatmap = ops.reshape(heatmap, (batch_size, num_joints, -1))
        idx, maxvals = ops.max(heatmap, axis=2, keep_dims=True)

        preds = ops.cast(ops.tile(idx, (1, 1, 2)), ms.float32)

        preds[:, :, 0] = preds[:, :, 0] % width
        preds[:, :, 1] = ops.floor((preds[:, :, 1]) / width)

        pred_mask = ops.tile(
            ms.numpy.greater(maxvals, 0.0, dtype=ms.float32), (1, 1, 2)
        )

        preds *= pred_mask
        return preds, maxvals

    def _shift_coordinate(self, coords: Tensor, heatmap: Tensor) -> Tensor:
        """shift the coordinate by +- 0.25 pixel"""
        batch_size = coords.shape[0]
        num_joints = coords.shape[1]
        heatmap_height = heatmap.shape[2]
        heatmap_width = heatmap.shape[3]

        int_coords = ops.cast(ops.round(coords), ms.int32)

        n = 0
        p = 0
        while n < batch_size:
            while p < num_joints:
                hm = heatmap[n][p]
                px = int_coords[n][p][0]
                py = int_coords[n][p][1]
                if (
                    px > 1
                    and px < heatmap_width - 1
                    and py > 1
                    and py < heatmap_height - 1
                ):
                    diff_x = hm[py][px + 1] - hm[py][px - 1]
                    diff_y = hm[py + 1][px] - hm[py - 1][px]
                    coords[n][p][0] += ms.numpy.sign(diff_x) * 0.25
                    coords[n][p][1] += ms.numpy.sign(diff_y) * 0.25
                p += 1
            n += 1
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

from typing import Tuple

import mindspore as ms
import mindspore.ops as ops

import numpy as np
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
            value. Default: False
        use_udp: Use Unbiased Data Processing (UDP) decoding. Default: False
        dark_udp_refine: Use post-refinement based on DARK / UDP. It cannot be
            use with `shift_coordinate` in the same time. Default: False
        kernel_size: Gaussian kernel size for UDP post-refinement, it should match
            the heatmap gaussian simg in training. K=17 for sigma=3 and
            K=11 for sigma=2. Default: 11

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
        shift_coordinate: bool = False,
        use_udp: bool = False,
        dark_udp_refine: bool = False,
        kernel_size: int = 11,
    ) -> None:
        super().__init__()
        self.pixel_std = pixel_std
        self.to_original = to_original
        self.shift_coordinate = shift_coordinate
        self.use_udp = use_udp
        self.dark_udp_refine = dark_udp_refine
        self.kernel_size = kernel_size

        if self.dark_udp_refine and self.shift_coordinate:
            raise ValueError(
                "`udp_refine` and `shift_coordinate` "
                "cannot be `true` in the same time."
            )

        if self.dark_udp_refine:
            self.gaussian_kernel = self._create_gaussian_kernel(self.kernel_size)
        else:
            self.gaussian_kernel = None

    def construct(
        self, heatmap: Tensor, center: Tensor, scale: Tensor, score: Tensor
    ) -> Tuple[Tensor, Tensor]:
        batch_size = heatmap.shape[0]

        coords, maxvals, maxvals_mask = self._get_max_preds(heatmap)
        if self.shift_coordinate:
            coords = self._shift_coordinate(coords, heatmap, maxvals_mask)
        elif self.dark_udp_refine:
            coords = self._dark_udp_refine_coords(coords, heatmap)
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

        # a boolean mask storing the location of the maximum value
        maxvals_mask = ops.zeros(heatmap.shape, ms.bool_)
        maxvals_mask = ops.tensor_scatter_elements(
            maxvals_mask, idx, ops.ones(idx.shape, ms.bool_), axis=2
        )
        maxvals_mask = ops.reshape(maxvals_mask, (batch_size, num_joints, -1, width))

        preds = ops.cast(ops.tile(idx, (1, 1, 2)), ms.float32)

        preds[:, :, 0] = preds[:, :, 0] % width
        preds[:, :, 1] = ops.floor((preds[:, :, 1]) / width)

        return preds, maxvals, maxvals_mask

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

    def _dark_udp_refine_coords(self, coords: Tensor, heatmap: Tensor) -> Tensor:
        """Refine the coords by moduluated heatmap"""
        N, K, H, W = heatmap.shape
        kernel = ops.tile(self.gaussian_kernel, (heatmap.shape[1], 1, 1, 1))
        heatmap = ops.conv2d(heatmap, kernel, group=heatmap.shape[1], pad_mode="same")
        heatmap = ops.clip_by_value(heatmap, 0.001, 50)
        heatmap = ops.log(heatmap)
        heatmap = ops.pad(heatmap, ((0, 0), (0, 0), (1, 1), (1, 1)))
        heatmap = heatmap.flatten()

        index = coords[..., 0] + 1 + (coords[..., 1] + 1) * (W + 2)
        index += (W + 2) * (H + 2) * ms.numpy.arange(0, N * K, 1).reshape(-1, K)
        index = ops.cast(index, ms.int32).reshape(-1, 1)
        i_ = heatmap[index]
        ix1 = heatmap[index + 1]
        iy1 = heatmap[index + W + 2]
        ix1y1 = heatmap[index + W + 3]
        ix1_y1_ = heatmap[index - W - 3]
        ix1_ = heatmap[index - 1]
        iy1_ = heatmap[index - 2 - W]

        dx = 0.5 * (ix1 - ix1_)
        dy = 0.5 * (iy1 - iy1_)
        derivative = ops.concat([dx, dy], axis=1)
        derivative = derivative.reshape(N, K, 2, 1)

        dxx = ix1 - 2 * i_ + ix1_
        dyy = iy1 - 2 * i_ + iy1_
        dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
        hessian = ops.concat([dxx, dxy, dxy, dyy], axis=1)
        hessian = hessian.reshape(N, K, 2, 2)

        hessian = ops.MatrixInverse()(hessian + ms.numpy.eye(2) * 1e-7)
        coords -= ops.Einsum("ijmn,ijnk->ijmk")((hessian, derivative)).squeeze()
        return coords

    def _create_gaussian_kernel(self, kernel_size: int) -> Tensor:
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
        xs = np.arange(-(kernel_size - 1) // 2, (kernel_size - 1) // 2 + 1, 1)
        ys = xs[:, None]
        kernel = np.exp(-(xs**2 + ys**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        kernel = kernel[None, None, ...]
        kernel = ms.Tensor(kernel, dtype=ms.float32)
        return kernel

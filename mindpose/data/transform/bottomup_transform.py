from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ...register import register

from ..column_names import COLUMN_MAP
from .transform import Transform
from .utils import fliplr_joints, get_affine_transform, pad_to_same, warp_affine_joints

__all__ = [
    "BottomUpTransform",
    "BottomUpHorizontalRandomFlip",
    "BottomUpRandomAffine",
    "BottomUpGenerateTarget",
    "BottomUpRescale",
    "BottomUpPad",
]

# set thread limitation
cv2.setNumThreads(2)


class BottomUpTransform(Transform):
    """Transform the input data into the output data based on bottom-up approach.

    Args:
        is_train: Whether the transformation is for training/testing. Default: True
        config: Method-specific configuration.  Default: None

    Inputs:
        data: Data tuples need to be transformed

    Outputs:
        result: Transformed data tuples

    Note:
        This is an abstract class, child class must implement `transform` method.
    """

    def setup_required_field(self) -> List[str]:
        """Get the required columns names used for this transformation.
        The columns names will be later used with Minspore Dataset `map` func.

        Returns:
            The column names
        """
        if self.is_train:
            return COLUMN_MAP["bottomup"]["train"]
        return COLUMN_MAP["bottomup"]["val"]

    def load_transform_cfg(self) -> Dict[str, Any]:
        """Loading the transform config, where the returned the config must
        be a dictionary which stores the configuration of this transformation,
        such as the transformed image size, etc.

        Returns:
            Transform configuration
        """
        transform_cfg = dict()

        transform_cfg["image_size"] = np.array(self.config["image_size"])
        transform_cfg["max_image_size"] = np.array(self.config["max_image_size"])
        transform_cfg["heatmap_sizes"] = np.array(self.config["heatmap_sizes"])
        assert len(transform_cfg["image_size"]) == 2
        for x in transform_cfg["heatmap_sizes"]:
            assert len(x) == 2

        # processing flip_pairs
        flip_pairs = np.array(self.config["flip_pairs"])
        if len(flip_pairs.shape) == 2:
            flip_index = flip_pairs[:, ::-1].flatten()
            flip_index = np.insert(flip_index, 0, 0)
        else:
            flip_index = flip_pairs

        transform_cfg["flip_pairs"] = flip_pairs
        transform_cfg["flip_index"] = flip_index

        transform_cfg["pixel_std"] = float(self.config["pixel_std"])
        transform_cfg["tag_per_joint"] = self.config["tag_per_joint"]

        return transform_cfg


@register("transform", extra_name="bottomup_horizontal_random_flip")
class BottomUpHorizontalRandomFlip(BottomUpTransform):
    """Perform randomly horizontal flip in bottomup approach.

    Args:
        is_train: Whether the transformation is for training/testing. Default: True
        config: Method-specific configuration. Default: None
        flip_prob: Probability of performing a horizontal flip. Default: 0.5
    """

    def __init__(
        self,
        is_train: bool = True,
        config: Optional[Dict[str, Any]] = None,
        flip_prob: float = 0.5,
    ) -> None:
        super().__init__(is_train, config)
        self.flip_prob = flip_prob

    def transform(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Transform the state into the transformed state. state is a dictionay
        storing the informaton of the image and labels, the returned states is
        the updated dictionary storing the updated image and labels.

        Args:
            state: Stored information of image and labels

        Returns:
            Updated inforamtion of image and labels based on the transformation

        Note:
            | Required `keys` for transform: image, mask, keypoints
            | Returned `keys` after transform: image, mask, keypoints
        """
        image = state["image"]
        keypoints = state["keypoints"]
        mask = state["mask"]

        heatmap_sizes = self._transform_cfg["heatmap_sizes"]

        if np.random.rand() <= self.flip_prob:
            image = cv2.flip(image, 1)
            for i, heatmap_size in enumerate(heatmap_sizes):
                width, height = heatmap_size
                patch_mask = mask[i, :height, :width]
                mask[i, :height, :width] = patch_mask[:, ::-1]

                keypoints[i] = fliplr_joints(
                    keypoints[i], width, flip_index=self._transform_cfg["flip_index"]
                )

        transformed_state = dict(image=image, keypoints=keypoints, mask=mask)
        return transformed_state


@register("transform", extra_name="bottomup_rescale")
class BottomUpRescale(BottomUpTransform):
    """Rescaling the image to the `max_image_size` without change the aspect ratio.

    Args:
        is_train: Whether the transformation is for training/testing. Default: True
        config: Method-specific configuration. Default: None
    """

    def _get_new_size(
        self, image_size: Tuple[int, int], max_size: Tuple[int, int]
    ) -> Tuple[int, int]:
        w, h = image_size
        max_w, max_h = max_size

        if w / h > max_w / max_h:
            target_w = max_w
            target_h = round(h * max_w / w)
        else:
            target_h = max_h
            target_w = round(w * max_h / h)

        return int(target_w), int(target_h)

    def transform(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Transform the state into the transformed state. state is a dictionay
        storing the informaton of the image and labels, the returned states is
        the updated dictionary storing the updated image and labels.

        Args:
            state: Stored information of image and labels

        Returns:
            Updated inforamtion of image and labels based on the transformation

        Note:
            | Required `keys` for transform: image
            | Returned `keys` after transform: image, center, scale, image_shape
        """
        image = state["image"]
        height, width = image.shape[:2]

        img_size = [width, height]
        target_size = self._get_new_size(
            img_size, self._transform_cfg["max_image_size"]
        )

        image = cv2.resize(
            image,
            (target_size[0], target_size[1]),
            interpolation=cv2.INTER_LINEAR,
        )

        pixel_std = self._transform_cfg["pixel_std"]
        center = np.array([round(width / 2), round(height / 2)])
        scale = np.array([width / pixel_std, height / pixel_std])

        transformed_state = dict()
        transformed_state["image"] = image
        transformed_state["center"] = center
        transformed_state["scale"] = scale
        transformed_state["image_shape"] = target_size
        return transformed_state


@register("transform", extra_name="bottomup_random_affine")
class BottomUpRandomAffine(BottomUpTransform):
    """Random affine transform the image. The mask and keypoints will be rescaled
    to the heatmap sizes after the transformation.

    Args:
        is_train: Whether the transformation is for training/testing. Default: True
        config: Method-specific configuration. Default: None
        rot_factor: Randomly rotated in [-rotation_factor, rotation_factor].
            Default: 30.
        scale_factor: Randomly Randomly scaled in [scale_factor[0], scale_factor[1]].
            Default: (0.75, 1.5)
        scale_type: Scaling with the long / short length of the image. Default: short
        trans_factor: Translation factor. Default: 40.
    """

    def __init__(
        self,
        is_train: bool = True,
        config: Optional[Dict[str, Any]] = None,
        rot_factor: float = 30.0,
        scale_factor: Tuple[float, float] = (0.75, 1.5),
        scale_type: str = "short",
        trans_factor: float = 40.0,
    ):

        super().__init__(is_train=is_train, config=config)
        self.max_rotation = rot_factor
        self.min_scale = scale_factor[0]
        self.max_scale = scale_factor[1]
        self.scale_type = scale_type
        self.trans_factor = trans_factor

    def _get_scale(
        self, image_size: Tuple[int, int], resized_size: Tuple[int, int]
    ) -> np.ndarray:
        w, h = image_size
        w_resized, h_resized = resized_size
        if w / w_resized < h / h_resized:
            if self.scale_type == "long":
                w_pad = h / h_resized * w_resized
                h_pad = h
            elif self.scale_type == "short":
                w_pad = w
                h_pad = w / w_resized * h_resized
            else:
                raise ValueError(f"Unknown scale type: {self.scale_type}")
        else:
            if self.scale_type == "long":
                w_pad = w
                h_pad = w / w_resized * h_resized
            elif self.scale_type == "short":
                w_pad = h / h_resized * w_resized
                h_pad = h
            else:
                raise ValueError(f"Unknown scale type: {self.scale_type}")

        scale = np.array([w_pad, h_pad], dtype=np.float32)

        return scale

    def transform(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Transform the state into the transformed state. state is a dictionay
        storing the informaton of the image and labels, the returned states is
        the updated dictionary storing the updated image and labels.

        Args:
            state: Stored information of image and labels

        Returns:
            Updated inforamtion of image and labels based on the transformation

        Note:
            | Required `keys` for transform: image, mask, keypoints
            | Returned `keys` after transform: image, mask, keypoints
        """
        image = state["image"]
        mask = state["mask"]
        keypoints = state["keypoints"]

        image_size = self._transform_cfg["image_size"]
        heatmap_sizes = self._transform_cfg["heatmap_sizes"]
        num_levels = len(heatmap_sizes)

        # expand the mask and keypoints for num_levels
        mask = [mask.copy() for _ in range(num_levels)]
        keypoints = [keypoints.copy() for _ in range(num_levels)]

        height, width = image.shape[:2]

        center = np.array((width / 2, height / 2))
        img_scale = np.array([width, height], dtype=np.float32)
        aug_scale = np.random.uniform(self.min_scale, self.max_scale)

        img_scale *= aug_scale
        aug_rot = np.random.uniform(-self.max_rotation, self.max_rotation)

        pixel_std = self._transform_cfg["pixel_std"]

        if self.trans_factor > 0:
            dx = np.random.randint(
                -self.trans_factor * img_scale[0] / pixel_std,
                self.trans_factor * img_scale[0] / pixel_std,
            )
            dy = np.random.randint(
                -self.trans_factor * img_scale[1] / pixel_std,
                self.trans_factor * img_scale[1] / pixel_std,
            )

            center[0] += dx
            center[1] += dy

        transformed_state = dict()

        for i, heatmap_size in enumerate(heatmap_sizes):
            # calculate the mat for heatmap
            scale = self._get_scale(img_scale, heatmap_size)
            mat = get_affine_transform(
                center=center,
                scale=scale / pixel_std,
                rot=aug_rot,
                output_size=heatmap_size,
                pixel_std=pixel_std,
            )

            # warp masks and keypoints
            mask[i] = cv2.warpAffine(
                mask[i],
                mat,
                (int(heatmap_size[0]), int(heatmap_size[1])),
                flags=cv2.INTER_NEAREST,
            )

            keypoints[i][:, :, 0:2] = warp_affine_joints(keypoints[i][:, :, 0:2], mat)

        # calculate the mat for image
        scale = self._get_scale(img_scale, image_size)
        mat = get_affine_transform(
            center=center,
            scale=scale / pixel_std,
            rot=aug_rot,
            output_size=image_size,
            pixel_std=pixel_std,
        )

        # warp image
        image = cv2.warpAffine(
            image,
            mat,
            (int(image_size[0]), int(image_size[1])),
            flags=cv2.INTER_LINEAR,
        )

        mask = pad_to_same(mask)

        transformed_state["image"] = image
        transformed_state["mask"] = mask
        transformed_state["keypoints"] = keypoints

        return transformed_state


@register("transform", extra_name="bottomup_generate_target")
class BottomUpGenerateTarget(BottomUpTransform):
    """Generate heatmap with the keypoint coordinatess with multiple scales.

    Args:
        is_train: Whether the transformation is for training/testing. Default: True
        config: Method-specific configuration. Default: None
        sigma: The sigmal size of gausian distribution. Default: 2.0
        max_num: Maximum number of instances within the image. Default: 30
        with_tag_mask: Output the tag mask for each resolution. Default: [True, False]

    Inputs:
        | data: Data tuples need to be transformed

    Outputs:
        | result: Transformed data tuples
    """

    def __init__(
        self,
        is_train: bool = True,
        config: Optional[Dict[str, Any]] = None,
        sigma: float = 2.0,
        max_num: int = 30,
        with_tag_mask: List[bool] = [True, False],
    ) -> None:
        super().__init__(is_train=is_train, config=config)
        self.sigma = sigma
        self.max_num = max_num
        self.with_tag_mask = with_tag_mask

    def transform(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Transform the state into the transformed state. state is a dictionay
        storing the informaton of the image and labels, the returned states is
        the updated dictionary storing the updated image and labels.

        Args:
            state: Stored information of image and labels

        Returns:
            Updated inforamtion of image and labels based on the transformation

        Note:
            | Required `keys` for transform: keypoints
            | Returned `keys` after transform: target, tag_mask
        """
        return self._encoding(state)

    def _encoding(self, state: Dict[str, Any]) -> Dict[str, Any]:
        target_list, tag_mask_list = list(), list()
        for i, (keypoint, heatmap_size) in enumerate(
            zip(state["keypoints"], self._transform_cfg["heatmap_sizes"])
        ):
            target, tag_mask = self._generate_heatmap_and_tag_mask(
                keypoint, heatmap_size, return_tag_mask=self.with_tag_mask[i]
            )

            target_list.append(target)
            if tag_mask is not None:
                tag_mask_list.append(tag_mask)

        # pad the heatmap to the same shape
        target_list = pad_to_same(target_list)
        if len(tag_mask_list) > 1:
            tag_mask_list = pad_to_same(tag_mask_list)
            tag_mask = np.stack(tag_mask_list)
        else:
            tag_mask = tag_mask_list[0]
            tag_mask = tag_mask[None, ...]

        tag_mask = np.stack(tag_mask)
        target = np.stack(target_list)

        transformed_state = dict(target=target, tag_mask=tag_mask)
        return transformed_state

    def _generate_heatmap_and_tag_mask(
        self,
        keypoints: np.ndarray,
        heatmap_size: np.ndarray,
        return_tag_mask: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate heatmap and tag mask for keypoints"""
        W, H = heatmap_size
        M, K, _ = keypoints.shape

        if M > self.max_num:
            raise ValueError(
                f"Number of keypoints in one image `{M}` "
                "exeeds the maximum num: `{self.max_num}`"
            )

        target = np.zeros((K, H, W), dtype=np.float32)
        if return_tag_mask:
            if self._transform_cfg["tag_per_joint"]:
                tag_mask = np.zeros((self.max_num, K, H, W), dtype=np.uint8)
            else:
                tag_mask = np.zeros((self.max_num, H, W), dtype=np.uint8)
        else:
            tag_mask = None

        # 3-sigma rule
        tmp_size = self.sigma * 3

        # gaussian kernel
        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, None]
        x0, y0 = size // 2, size // 2
        # The gaussian is not normalized,
        # we want the center value to equal 1
        g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma**2))

        for m, single_keypoints in enumerate(keypoints):
            for idx, pt in enumerate(single_keypoints):
                if pt[2] > 0:
                    mu_x, mu_y = round(pt[0]), round(pt[1])

                    # Check that any part of the gaussian is in-bounds
                    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                    if ul[0] >= W or ul[1] >= H or br[0] < 0 or br[1] < 0:
                        continue

                    # Usable gaussian range
                    g_x = max(0, -ul[0]), min(br[0], W) - ul[0]
                    g_y = max(0, -ul[1]), min(br[1], H) - ul[1]
                    # Image range
                    img_x = max(0, ul[0]), min(br[0], W)
                    img_y = max(0, ul[1]), min(br[1], H)

                    heatmap_patch = target[
                        idx, img_y[0] : img_y[1], img_x[0] : img_x[1]
                    ]

                    target[idx, img_y[0] : img_y[1], img_x[0] : img_x[1]] = np.maximum(
                        heatmap_patch,
                        g[g_y[0] : g_y[1], g_x[0] : g_x[1]],
                    )

                    if mu_x >= W or mu_y >= H or mu_x < 0 or mu_y < 0:
                        continue

                    if return_tag_mask:
                        if self._transform_cfg["tag_per_joint"]:
                            tag_mask[m, idx, mu_y, mu_x] = True
                        else:
                            tag_mask[m, mu_y, mu_x] = True

        return target, tag_mask


@register("transform", extra_name="bottomup_pad")
class BottomUpPad(BottomUpTransform):
    """Padding the image to the `max_image_size`.

    Args:
        is_train: Whether the transformation is for training/testing. Default: True
        config: Method-specific configuration. Default: None
    """

    def transform(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Transform the state into the transformed state. state is a dictionay
        storing the informaton of the image and labels, the returned states is
        the updated dictionary storing the updated image and labels.

        Args:
            state: Stored information of image and labels

        Returns:
            Updated inforamtion of image and labels based on the transformation

        Note:
            | Required `keys` for transform: image
            | Returned `keys` after transform: image, mask
        """
        image = state["image"]
        height, width = image.shape[:2]
        target_width, target_height = self._transform_cfg["max_image_size"]
        assert target_width >= width
        assert target_height >= height

        height_pad = target_height - height
        width_pad = target_width - width
        image = np.pad(image, ((0, height_pad), (0, width_pad), (0, 0)))

        mask = np.zeros((target_height, target_width), dtype=np.uint8)
        mask[:height, :width] = 1

        transformed_state = dict()
        transformed_state["image"] = image
        transformed_state["mask"] = mask
        return transformed_state

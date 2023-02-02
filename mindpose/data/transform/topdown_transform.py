from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ...register import register

from ..column_names import COLUMN_MAP
from .transform import Transform
from .utils import affine_transform, fliplr_joints, get_affine_transform

__all__ = [
    "TopDownTransform",
    "TopDownAffineToSingle",
    "TopDownGenerateTarget",
    "TopDownHorizontalRandomFlip",
    "TopDownHalfBodyTransform",
    "TopDownRandomScaleRotation",
]


class TopDownTransform(Transform):
    """Transform the input data into the output data based on top-down approach.
    This is an abstract class, child class must implement `transform` method.

    Args:
        is_train: Whether the transformation is for training/testing. Default: True
        config: Method-specific configuration. Default: None

    Inputs:
        data: Data tuples need to be transformed

    Outputs:
        result: Transformed data tuples
    """

    def setup_required_field(self) -> List[str]:
        if self.is_train:
            return COLUMN_MAP["topdown"]["train"]
        return COLUMN_MAP["topdown"]["val"]

    def load_transform_cfg(self) -> Dict[str, Any]:
        """Loading the annoation info from the config file"""
        transform_cfg = dict()
        transform_cfg["image_size"] = np.array(self.config["image_size"])
        transform_cfg["heatmap_size"] = np.array(self.config["heatmap_size"])
        assert len(transform_cfg["image_size"]) == 2
        assert len(transform_cfg["heatmap_size"]) == 2

        transform_cfg["joint_weights"] = np.array(self.config["joint_weights"])
        transform_cfg["flip_pairs"] = np.array(self.config["flip_pairs"])
        transform_cfg["upper_body_ids"] = np.array(self.config["upper_body_ids"])
        transform_cfg["pixel_std"] = float(self.config["pixel_std"])

        return transform_cfg


@register("transform", extra_name="topdown_affine_to_single")
class TopDownAffineToSingle(TopDownTransform):
    """Affine transform the image to output cropped images with single instance.

    Args:
        is_train: Whether the transformation is for training/testing. Default: True
        config: Method-specific configuration. Default: None

    Inputs:
        data: Data tuples need to be transformed.

    Outputs:
        result: Transformed data tuples
    """

    def transform(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Perform affine transform

        Required `keys` in `state`: image, center, scale, rotation, keypoints (optional)
        Output `keys` in transformed state`: image, keypoints (optional)
        """
        image_size = self._transform_cfg["image_size"]
        pixel_std = self._transform_cfg["pixel_std"]

        trans = get_affine_transform(
            state["center"],
            state["scale"],
            state["rotation"],
            image_size,
            pixel_std=pixel_std,
        )

        transformed_state = dict()

        transformed_state["image"] = cv2.warpAffine(
            state["image"],
            trans,
            (int(image_size[0]), int(image_size[1])),
            flags=cv2.INTER_LINEAR,
        )

        if "keypoints" in state:
            num_joints = state["keypoints"].shape[0]
            for i in range(num_joints):
                transformed_state["keypoints"] = state["keypoints"]
                if transformed_state["keypoints"][i, 2] > 0.0:
                    transformed_state["keypoints"][i, 0:2] = affine_transform(
                        transformed_state["keypoints"][i, 0:2], trans
                    )

        return transformed_state


@register("transform", extra_name="topdown_generate_target")
class TopDownGenerateTarget(TopDownTransform):
    """Generate heatmap from the coordinates

    Args:
        is_train: Whether the transformation is for training/testing. Default: True
        config: Method-specific configuration. Default: None
        sigma: The sigmal size of gausian distribution. Default: 2.0
        use_different_joint_weights: Use extra joint weight in target weight calculation. Default: False
        subpixel_center: When true, the center of the heatmap is in subpixel-level.
            Otherwise, the center is rounded to nearest pixel. Default: False

    Inputs:
        data: Data tuples need to be transformed.

    Outputs:
        result: Transformed data tuples
    """

    def __init__(
        self,
        is_train: bool = True,
        config: Optional[Dict[str, Any]] = None,
        sigma: float = 2.0,
        use_different_joint_weights: bool = False,
        subpixel_center: bool = False,
    ) -> None:
        super().__init__(is_train=is_train, config=config)
        self.sigma = sigma
        self.use_different_joint_weights = use_different_joint_weights
        self.subpixel_center = subpixel_center

    def transform(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate the heatmap

        Required `keys` in `state`: keypoints
        Output `keys` in transformed state`: taget, target_weight
        """

        image_size = self._transform_cfg["image_size"]
        W, H = self._transform_cfg["heatmap_size"]
        joint_weights = self._transform_cfg["joint_weights"]
        keypoints = state["keypoints"]

        num_joints = keypoints.shape[0]
        target_weight = np.zeros((num_joints, 1), dtype=np.float32)
        target = np.zeros((num_joints, H, W), dtype=np.float32)

        # 3-sigma rule
        tmp_size = self.sigma * 3

        for joint_id in range(num_joints):
            target_weight[joint_id] = keypoints[joint_id, 2]

            feat_stride = image_size / np.array([W, H])
            mu_x = keypoints[joint_id][0] / feat_stride[0]
            mu_y = keypoints[joint_id][1] / feat_stride[1]
            if not self.subpixel_center:
                mu_x, mu_y = round(mu_x), round(mu_y)

            # Check that any part of the gaussian is in-bounds
            ul = [mu_x - tmp_size, mu_y - tmp_size]
            br = [mu_x + tmp_size, mu_y + tmp_size]
            if ul[0] >= W or ul[1] >= H or br[0] < 0 or br[1] < 0:
                target_weight[joint_id] = 0

            if target_weight[joint_id] > 0.5:
                x = np.arange(W, dtype=np.float32)
                y = np.arange(H, dtype=np.float32)[:, None]
                g = np.exp(-((x - mu_x) ** 2 + (y - mu_y) ** 2) / (2 * self.sigma**2))
                target[joint_id] = g

        if self.use_different_joint_weights:
            target_weight = np.multiply(target_weight, joint_weights)

        transformed_state = dict(target=target, target_weight=target_weight.squeeze())
        return transformed_state


@register("transform", extra_name="topdown_horizontal_random_flip")
class TopDownHorizontalRandomFlip(TopDownTransform):
    """Perform randomly horizontal flip

    Args:
        is_train: Whether the transformation is for training/testing. Default: True
        config: Method-specific configuration. Default: None
        flip_prob: Probability of performing a horizontal flip. Default: 0.5

    Inputs:
        data: Data tuples need to be transformed.

    Outputs:
        result: Transformed data tuples
    """

    def __init__(
        self,
        is_train: bool = True,
        config: Optional[Dict[str, Any]] = None,
        flip_prob: float = 0.5,
    ) -> None:
        super().__init__(is_train=is_train, config=config)
        self.flip_prob = flip_prob

    def transform(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Perform randomly horizontal flip

        Required `keys` in `state`: image, keypoints, center
        Output `keys` in transformed state`: image, keypoints, center
        """
        image = state["image"]
        keypoints = state["keypoints"]
        center = state["center"]

        if np.random.rand() <= self.flip_prob:
            image = image[:, ::-1, :]
            keypoints = fliplr_joints(
                keypoints,
                image.shape[1],
                self._transform_cfg["flip_pairs"],
            )
            center[0] = image.shape[1] - center[0]

        transformed_state = dict(image=image, keypoints=keypoints, center=center)
        return transformed_state


@register("transform", extra_name="topdown_halfbody_transform")
class TopDownHalfBodyTransform(TopDownTransform):
    """Perform half-body transform. Keep only the upper body or the lower body at random.

    Args:
        is_train: Whether the transformation is for training/testing. Default: True
        config: Method-specific configuration. Default: None
        num_joints_half_body: Threshold number of performing half-body transform. Default: 8
        prob_half_body: Probability of performing half-body transform. Default: 0.3
        scale_padding: Extra scale padding multiplier in generating the cropped images. Default: 1.5

    Inputs:
        data: Data tuples need to be transformed.

    Outputs:
        result: Transformed data tuples
    """

    def __init__(
        self,
        is_train: bool = True,
        config: Optional[Dict[str, Any]] = None,
        num_joints_half_body: int = 8,
        prob_half_body: float = 0.3,
        scale_padding: float = 1.5,
    ) -> None:
        super().__init__(is_train=is_train, config=config)
        self.num_joints_half_body = num_joints_half_body
        self.prob_half_body = prob_half_body
        self.scale_padding = scale_padding

    def half_body_transform(
        self, keypoints: np.ndarray, num_joints: int = 17
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get center&scale for half-body transform."""
        upper_joints = []
        lower_joints = []
        for joint_id in range(num_joints):
            if keypoints[joint_id][2] > 0:
                if joint_id in self._transform_cfg["upper_body_ids"]:
                    upper_joints.append(keypoints[joint_id])
                else:
                    lower_joints.append(keypoints[joint_id])

        if np.random.randn() < 0.5 and len(upper_joints) > 2:
            selected_joints = upper_joints
        elif len(lower_joints) > 2:
            selected_joints = lower_joints
        else:
            selected_joints = upper_joints

        if len(selected_joints) < 2:
            return None, None

        selected_joints = np.array(selected_joints, dtype=np.float32)
        center = selected_joints.mean(axis=0)[:2]

        left_top = np.amin(selected_joints, axis=0)

        right_bottom = np.amax(selected_joints, axis=0)

        w = right_bottom[0] - left_top[0]
        h = right_bottom[1] - left_top[1]

        aspect_ratio = (
            self._transform_cfg["image_size"][0] / self._transform_cfg["image_size"][1]
        )

        if w > aspect_ratio * h:
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio

        scale = np.array(
            [
                w / self._transform_cfg["pixel_std"],
                h / self._transform_cfg["pixel_std"],
            ],
            dtype=np.float32,
        )
        scale = scale * self.scale_padding
        return center, scale

    def transform(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Perform half-body transform.

        Required `keys` in `state`: keypoints
        Output `keys` in transformed state`: center, scale
        """
        keypoints = state["keypoints"]
        num_joints = keypoints.shape[0]

        if (
            np.sum(keypoints[:, 2]) > self.num_joints_half_body
            and np.random.rand() < self.prob_half_body
        ):

            c_half_body, s_half_body = self.half_body_transform(
                keypoints, num_joints=num_joints
            )

            if c_half_body is not None and s_half_body is not None:
                transformed_state = dict(center=c_half_body, scale=s_half_body)
                return transformed_state

        return dict()


@register("transform", extra_name="topdown_randomscale_rotation")
class TopDownRandomScaleRotation(TopDownTransform):
    """Perform random scaling and rotations

    Args:
        is_train: Whether the transformation is for training/testing. Default: True
        config: Method-specific configuration. Default: None
        rot_factor: Std of rotation degree. Default: 40.
        scale_factor: Std of scaling value. Default: 0.5
        rot_prob: Probability of performing rotation. Default: 0.6

    Inputs:
        data: Data tuples need to be transformed.

    Outputs:
        result: Transformed data tuples
    """

    def __init__(
        self,
        is_train: bool = True,
        config: Optional[Dict[str, Any]] = None,
        rot_factor: float = 40.0,
        scale_factor: float = 0.5,
        rot_prob: float = 0.6,
    ) -> None:
        super().__init__(is_train=is_train, config=config)
        self.rot_factor = rot_factor
        self.scale_factor = scale_factor
        self.rot_prob = rot_prob

    def transform(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Perform random rotation and scaling

        Required `keys` in `state`: "scale"
        Output `keys` in transformed state`: scale, rotation
        """
        s = state["scale"]

        sf = self.scale_factor
        rf = self.rot_factor

        s_factor = np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
        s = s * s_factor

        r_factor = np.clip(np.random.randn() * rf, -rf * 2, rf * 2)
        r = r_factor if np.random.rand() <= self.rot_prob else 0

        transformed_state = dict(scale=s, rotation=r)
        return transformed_state

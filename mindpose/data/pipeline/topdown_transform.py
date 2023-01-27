from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..column_names import COLUMN_MAP
from .transform import Transform
from .utils import affine_transform, fliplr_joints, get_affine_transform


class TopDownTransform(Transform):
    def setup_required_field(self) -> List[str]:
        if self.is_train:
            return COLUMN_MAP["topdown"]["train"]
        return COLUMN_MAP["topdown"]["val"]

    def load_transform_cfg(self) -> Dict[str, Any]:
        """Loading the annoation info from the config file"""
        transform_cfg = dict()
        transform_cfg["image_size"] = self.config.get("image_size", [192, 256])
        transform_cfg["heatmap_size"] = self.config.get("heatmap_size", [48, 64])

        # TODO: read array from config
        transform_cfg["joint_weights"] = [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.2,
            1.2,
            1.5,
            1.5,
            1.0,
            1.0,
            1.2,
            1.2,
            1.5,
            1.5,
        ]

        transform_cfg["flip_pairs"] = [
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
            [9, 10],
            [11, 12],
            [13, 14],
            [15, 16],
        ]

        transform_cfg["upper_body_ids"] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        return transform_cfg


class TopDownAffineToSingle(TopDownTransform):
    """Affine transform the image to make input with single instance."""

    def transform(self, state: Dict[str, Any]) -> Dict[str, Any]:
        image_size = self._transform_cfg["image_size"]

        trans = get_affine_transform(
            state["center"], state["scale"], state["rotation"], image_size
        )

        transformed_state = dict()

        transformed_state["image"] = cv2.warpAffine(
            state["image"],
            trans,
            (int(image_size[0]), int(image_size[1])),
            flags=cv2.INTER_LINEAR,
        )

        if "keypoints" in state:
            for i in range(self.num_joints):
                transformed_state["keypoints"] = state["keypoints"]
                if transformed_state["keypoints"][i, 2] > 0.0:
                    transformed_state["keypoints"][i, 0:2] = affine_transform(
                        transformed_state["keypoints"][i, 0:2], trans
                    )

        return transformed_state


class TopDownGenerateTarget(TopDownTransform):
    """Generate the target heatmap."""

    def __init__(
        self,
        is_train: bool = True,
        config: Optional[Dict[str, Any]] = None,
        num_joints: int = 17,
        sigma: float = 2.0,
        use_different_joint_weights: bool = False,
    ) -> None:
        super().__init__(is_train=is_train, num_joints=num_joints, config=config)
        self.sigma = sigma
        self.use_different_joint_weights = use_different_joint_weights

    def transform(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate the target heatmap via "MSRA" approach."""

        image_size = self._transform_cfg["image_size"]
        W, H = self._transform_cfg["heatmap_size"]
        joint_weights = self._transform_cfg["joint_weights"]

        target_weight = np.zeros((self.num_joints, 1), dtype=np.float32)
        target = np.zeros((self.num_joints, H, W), dtype=np.float32)

        # 3-sigma rule
        tmp_size = self.sigma * 3
        keypoints = state["keypoints"]

        for joint_id in range(self.num_joints):
            target_weight[joint_id] = keypoints[joint_id, 2]

            feat_stride = image_size / np.array([W, H])
            mu_x = int(keypoints[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(keypoints[joint_id][1] / feat_stride[1] + 0.5)
            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= W or ul[1] >= H or br[0] < 0 or br[1] < 0:
                target_weight[joint_id] = 0

            if target_weight[joint_id] > 0.5:
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, None]
                x0 = y0 = size // 2
                # The gaussian is not normalized,
                # we want the center value to equal 1
                g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma**2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], W) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], H) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], W)
                img_y = max(0, ul[1]), min(br[1], H)

                target[joint_id][img_y[0] : img_y[1], img_x[0] : img_x[1]] = g[
                    g_y[0] : g_y[1], g_x[0] : g_x[1]
                ]

        if self.use_different_joint_weights:
            target_weight = np.multiply(target_weight, joint_weights)

        transformed_state = dict(target=target, target_weight=target_weight.squeeze())
        return transformed_state


class TopDownHorizontalRandomFlip(TopDownTransform):
    """Data augmentation with random image horizontal flip.

    Args:
        flip_prob (float): Probability of flip.
    """

    def __init__(
        self,
        is_train: bool = True,
        config: Optional[Dict[str, Any]] = None,
        num_joints: int = 17,
        flip_prob: float = 0.5,
    ) -> None:
        super().__init__(is_train=is_train, config=config, num_joints=num_joints)
        self.flip_prob = flip_prob

    def transform(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Perform data augmentation with random image flip."""
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
            center[0] = image.shape[1] - center[0] - 1

        transformed_state = dict(image=image, keypoints=keypoints, center=center)
        return transformed_state


class TopDownHalfBodyTransform(TopDownTransform):
    """Data augmentation with half-body transform. Keep only the upper body or
    the lower body at random.

    Args:
        num_joints_half_body (int): Threshold of performing
            half-body transform. If the body has fewer number
            of joints (< num_joints_half_body), ignore this step.
        prob_half_body (float): Probability of half-body transform.
    """

    def __init__(
        self,
        is_train: bool = True,
        config: Optional[Dict[str, Any]] = None,
        num_joints: int = 17,
        num_joints_half_body: int = 8,
        prob_half_body: float = 0.3,
        pixel_std: float = 200.0,
        scale_padding: float = 1.5,
    ) -> None:
        super().__init__(is_train=is_train, config=config, num_joints=num_joints)
        self.num_joints_half_body = num_joints_half_body
        self.prob_half_body = prob_half_body
        self.pixel_std = float(pixel_std)
        self.scale_padding = scale_padding

    def half_body_transform(
        self, keypoints: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get center&scale for half-body transform."""
        upper_joints = []
        lower_joints = []
        for joint_id in range(self.num_joints):
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

        scale = np.array([w / self.pixel_std, h / self.pixel_std], dtype=np.float32)
        scale = scale * self.scale_padding
        return center, scale

    def transform(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Perform data augmentation with half-body transform."""
        keypoints = state["keypoints"]

        if (
            np.sum(keypoints[:, 2]) > self.num_joints_half_body
            and np.random.rand() < self.prob_half_body
        ):

            c_half_body, s_half_body = self.half_body_transform(keypoints)

            if c_half_body is not None and s_half_body is not None:
                state["center"] = c_half_body
                state["scale"] = s_half_body
                transformed_state = dict(center=c_half_body, scale=s_half_body)
                return transformed_state

        return dict()


class TopDownRandomScaleRotation(TopDownTransform):
    """Data augmentation with random scaling & rotating.

    Args:
        rot_factor (int): Rotating to ``[-2*rot_factor, 2*rot_factor]``.
        scale_factor (float): Scaling to ``[1-scale_factor, 1+scale_factor]``.
        rot_prob (float): Probability of random rotation.
    """

    def __init__(
        self,
        is_train: bool = True,
        config: Optional[Dict[str, Any]] = None,
        num_joints: int = 17,
        rot_factor: float = 40.0,
        scale_factor: float = 0.5,
        rot_prob: float = 0.6,
    ) -> None:
        super().__init__(is_train=is_train, config=config, num_joints=num_joints)
        self.rot_factor = rot_factor
        self.scale_factor = scale_factor
        self.rot_prob = rot_prob

    def transform(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Perform data augmentation with random scaling & rotating."""
        s = state["scale"]

        sf = self.scale_factor
        rf = self.rot_factor

        s_factor = np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
        s = s * s_factor

        r_factor = np.clip(np.random.randn() * rf, -rf * 2, rf * 2)
        r = r_factor if np.random.rand() <= self.rot_prob else 0

        transformed_state = dict(scale=s, rotation=r)
        return transformed_state


TOPDOWN_TRANSFORM_MAPPING = {
    "topdown_horizontal_random_flip": TopDownHorizontalRandomFlip,
    "topdown_halfbody_transform": TopDownHalfBodyTransform,
    "topdown_randomscale_rotation": TopDownRandomScaleRotation,
    "topdown_affine_to_single": TopDownAffineToSingle,
    "topdown_generate_target": TopDownGenerateTarget,
}

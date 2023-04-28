from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ...register import register

from ..column_names import COLUMN_MAP
from .transform import Transform
from .utils import (
    affine_transform,
    fliplr_joints,
    get_affine_transform,
    get_warp_matrix,
    warp_affine_joints,
)

__all__ = [
    "TopDownTransform",
    "TopDownBoxToCenterScale",
    "TopDownAffine",
    "TopDownGenerateTarget",
    "TopDownHorizontalRandomFlip",
    "TopDownHalfBodyTransform",
    "TopDownRandomScaleRotation",
]

# set thread limitation
cv2.setNumThreads(2)


class TopDownTransform(Transform):
    """Transform the input data into the output data based on top-down approach.

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
            return COLUMN_MAP["topdown"]["train"]
        return COLUMN_MAP["topdown"]["val"]

    def load_transform_cfg(self) -> Dict[str, Any]:
        """Loading the transform config, where the returned the config must
        be a dictionary which stores the configuration of this transformation,
        such as the transformed image size, etc.

        Returns:
            Transform configuration
        """
        transform_cfg = dict()
        transform_cfg["image_size"] = np.array(self.config["image_size"])
        transform_cfg["heatmap_size"] = np.array(self.config["heatmap_size"])
        assert len(transform_cfg["image_size"]) == 2
        assert len(transform_cfg["heatmap_size"]) == 2

        # processing flip_pairs
        flip_pairs = np.array(self.config["flip_pairs"])
        if len(flip_pairs.shape) == 2:
            flip_index = flip_pairs[:, ::-1].flatten()
            flip_index = np.insert(flip_index, 0, 0)
        else:
            flip_index = flip_pairs

        transform_cfg["flip_pairs"] = flip_pairs
        transform_cfg["flip_index"] = flip_index

        transform_cfg["upper_body_ids"] = np.array(self.config["upper_body_ids"])
        transform_cfg["pixel_std"] = float(self.config["pixel_std"])
        transform_cfg["scale_padding"] = float(self.config["scale_padding"])

        if "joint_weights" in self.config:
            transform_cfg["joint_weights"] = np.array(self.config["joint_weights"])
        else:
            transform_cfg["joint_weights"] = None

        return transform_cfg


@register("transform", extra_name="topdown_box_to_center_scale")
class TopDownBoxToCenterScale(TopDownTransform):
    """Convert the box coordinate to center and scale. If `is_train` is True,
    the center will be randomly shifted by a small amount.

    Args:
        is_train: Whether the transformation is for training/testing. Default: True
        config: Method-specific configuration. Default: None

    Inputs:
        | data: Data tuples need to be transformed

    Outputs:
        | result: Transformed data tuples
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
            | Required `keys` for transform: boxes
            | Returned `keys` after transform: center, scale
        """
        center, scale = self._xywh2cs(*state["boxes"])
        return dict(center=center, scale=scale)

    def _xywh2cs(
        self, x: float, y: float, w: float, h: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        aspect_ratio = (
            self._transform_cfg["image_size"][0] / self._transform_cfg["image_size"][1]
        )
        center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)

        # perform a random center shift for training dataset
        if self.is_train and np.random.rand() < 0.3:
            center += np.random.uniform(-0.2, 0.2, size=2) * [w, h]

        if w > aspect_ratio * h:
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio

        pixel_std = self._transform_cfg["pixel_std"]
        scale_padding = self._transform_cfg["scale_padding"]

        scale = np.array([w / pixel_std, h / pixel_std], dtype=np.float32)
        # padding to include proper amount of context
        scale = scale * scale_padding
        return center, scale


@register("transform", extra_name="topdown_affine")
class TopDownAffine(TopDownTransform):
    """Affine transform the image, and the transform image will
    contain single instance only.

    Args:
        is_train: Whether the transformation is for training/testing. Default: True
        config: Method-specific configuration. Default: None
        use_udp: Use Unbiased Data Processing (UDP) affine transform. Default: False

    Inputs:
        | data: Data tuples need to be transformed

    Outputs:
        | result: Transformed data tuples
    """

    def __init__(
        self,
        is_train: bool = True,
        config: Optional[Dict[str, Any]] = None,
        use_udp: bool = False,
    ) -> None:
        super().__init__(is_train=is_train, config=config)
        self.use_udp = use_udp

    def transform(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Transform the state into the transformed state. state is a dictionay
        storing the informaton of the image and labels, the returned states is
        the updated dictionary storing the updated image and labels.

        Args:
            state: Stored information of image and labels

        Returns:
            Updated inforamtion of image and labels based on the transformation

        Note:
            | Required `keys` for transform: image, center, scale, rotation,
                keypoints (optional)
            | Returned `keys` after transform: image, keypoints (optional)
        """
        if self.use_udp:
            return self._udp_affine(state)
        return self._affine(state)

    def _affine(self, state: Dict[str, Any]) -> Dict[str, Any]:
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

    def _udp_affine(self, state: Dict[str, Any]) -> Dict[str, Any]:
        image_size = self._transform_cfg["image_size"]
        pixel_std = self._transform_cfg["pixel_std"]

        trans = get_warp_matrix(
            state["rotation"],
            state["center"] * 2.0,
            image_size - 1.0,
            state["scale"] * pixel_std,
        )

        transformed_state = dict()

        transformed_state["image"] = cv2.warpAffine(
            state["image"],
            trans,
            (int(image_size[0]), int(image_size[1])),
            flags=cv2.INTER_LINEAR,
        )

        if "keypoints" in state:
            transformed_state["keypoints"] = state["keypoints"]
            transformed_state["keypoints"][:, 0:2] = warp_affine_joints(
                state["keypoints"][:, 0:2], trans
            )

        return transformed_state


@register("transform", extra_name="topdown_generate_target")
class TopDownGenerateTarget(TopDownTransform):
    """Generate heatmap from the coordinates.

    Args:
        is_train: Whether the transformation is for training/testing. Default: True
        config: Method-specific configuration. Default: None
        sigma: The sigmal size of gausian distribution. Default: 2.0
        use_different_joint_weights: Use extra joint weight in target weight
            calculation. Default: False
        use_udp: Use Unbiased Data Processing (UDP) encoding. Default: False

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
        use_different_joint_weights: bool = False,
        use_udp: bool = False,
    ) -> None:
        super().__init__(is_train=is_train, config=config)
        self.sigma = sigma
        self.use_different_joint_weights = use_different_joint_weights
        self.use_udp = use_udp

        if (
            self.use_different_joint_weights
            and self._transform_cfg["joint_weights"] is None
        ):
            raise ValueError(
                "`joint_weights` must be provided "
                "if `use_different_joint_weights` is True."
            )

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
            | Returned `keys` after transform: target, target_weight
        """
        if self.use_udp:
            return self._udp_encoding(state)
        return self._encoding(state)

    def _encoding(self, state: Dict[str, Any]) -> Dict[str, Any]:
        image_size = self._transform_cfg["image_size"]
        W, H = self._transform_cfg["heatmap_size"]
        joint_weights = self._transform_cfg["joint_weights"]
        keypoints = state["keypoints"]

        num_joints = keypoints.shape[0]
        target_weight = np.zeros(num_joints, dtype=np.float32)
        target = np.zeros((num_joints, H, W), dtype=np.float32)

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

        for joint_id in range(num_joints):
            target_weight[joint_id] = keypoints[joint_id, 2]

            feat_stride = image_size / np.array([W, H])
            mu_x = round(keypoints[joint_id][0] / feat_stride[0])
            mu_y = round(keypoints[joint_id][1] / feat_stride[1])
            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= W or ul[1] >= H or br[0] < 0 or br[1] < 0:
                target_weight[joint_id] = 0
                continue

            if target_weight[joint_id] > 0.5:
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

        transformed_state = dict(target=target, target_weight=target_weight)
        return transformed_state

    def _udp_encoding(self, state: Dict[str, Any]) -> Dict[str, Any]:
        image_size = self._transform_cfg["image_size"]
        W, H = self._transform_cfg["heatmap_size"]
        joint_weights = self._transform_cfg["joint_weights"]
        keypoints = state["keypoints"]

        num_joints = keypoints.shape[0]
        target_weight = np.zeros(num_joints, dtype=np.float32)
        target = np.zeros((num_joints, H, W), dtype=np.float32)

        # 3-sigma rule
        tmp_size = self.sigma * 3

        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, None]
        x0, y0 = size // 2, size // 2

        for joint_id in range(num_joints):
            target_weight[joint_id] = keypoints[joint_id, 2]

            feat_stride = (image_size - 1.0) / (np.array([W, H]) - 1.0)
            mu_x = int(keypoints[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(keypoints[joint_id][1] / feat_stride[1] + 0.5)
            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= W or ul[1] >= H or br[0] < 0 or br[1] < 0:
                target_weight[joint_id] = 0
                continue

            if target_weight[joint_id] > 0.5:
                mu_x_ac = keypoints[joint_id][0] / feat_stride[0]
                mu_y_ac = keypoints[joint_id][1] / feat_stride[1]
                x0_p = x0 + mu_x_ac - mu_x
                y0_p = y0 + mu_y_ac - mu_y
                g = np.exp(-((x - x0_p) ** 2 + (y - y0_p) ** 2) / (2 * self.sigma**2))

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

        transformed_state = dict(target=target, target_weight=target_weight)
        return transformed_state


@register("transform", extra_name="topdown_horizontal_random_flip")
class TopDownHorizontalRandomFlip(TopDownTransform):
    """Perform randomly horizontal flip in topdown approach.

    Args:
        is_train: Whether the transformation is for training/testing. Default: True
        config: Method-specific configuration. Default: None
        flip_prob: Probability of performing a horizontal flip. Default: 0.5

    Inputs:
        | data: Data tuples need to be transformed

    Outputs:
        | result: Transformed data tuples
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
        """Transform the state into the transformed state. state is a dictionay
        storing the informaton of the image and labels, the returned states is
        the updated dictionary storing the updated image and labels.

        Args:
            state: Stored information of image and labels

        Returns:
            Updated inforamtion of image and labels based on the transformation

        Note:
            | Required `keys` for transform: image, keypoints, center
            | Returned `keys` after transform: image, keypoints, center
        """
        image = state["image"]
        keypoints = state["keypoints"]
        center = state["center"]

        if np.random.rand() <= self.flip_prob:
            image = cv2.flip(image, 1)
            keypoints = fliplr_joints(
                keypoints,
                image.shape[1],
                flip_index=self._transform_cfg["flip_index"],
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
        num_joints_half_body: Threshold number of performing half-body transform.
            Default: 8
        prob_half_body: Probability of performing half-body transform. Default: 0.3
        scale_padding: Extra scale padding multiplier in generating the cropped images.
            Default: 1.5

    Inputs:
        | data: Data tuples need to be transformed

    Outputs:
        | result: Transformed data tuples
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
        # Get center and scale for half-body transform
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
        """Transform the state into the transformed state. state is a dictionay
        storing the informaton of the image and labels, the returned states is
        the updated dictionary storing the updated image and labels.

        Args:
            state: Stored information of image and labels

        Returns:
            Updated inforamtion of image and labels based on the transformation

        Note:
            | Required `keys` for transform: keypoints
            | Returned `keys` after transform: center, scale
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
    """Perform random scaling and rotation.

    Args:
        is_train: Whether the transformation is for training/testing. Default: True
        config: Method-specific configuration. Default: None
        rot_factor: Std of rotation degree. Default: 40.
        scale_factor: Std of scaling value. Default: 0.5
        rot_prob: Probability of performing rotation. Default: 0.6

    Inputs:
        | data: Data tuples need to be transformed

    Outputs:
        | result: Transformed data tuples
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
        """Transform the state into the transformed state. state is a dictionay
        storing the informaton of the image and labels, the returned states is
        the updated dictionary storing the updated image and labels.

        Args:
            state: Stored information of image and labels

        Returns:
            Updated inforamtion of image and labels based on the transformation

        Note:
            | Required `keys` for transform: scale
            | Returned `keys` after transform: scale, rotation
        """
        s = state["scale"]

        sf = self.scale_factor
        rf = self.rot_factor

        s_factor = np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf, dtype=np.float32)
        s = s * s_factor

        r_factor = np.clip(np.random.randn() * rf, -rf * 2, rf * 2, dtype=np.float32)
        r = r_factor if np.random.rand() <= self.rot_prob else np.float32(0.0)

        transformed_state = dict(scale=s, rotation=r)
        return transformed_state

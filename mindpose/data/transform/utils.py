from typing import List, Tuple

import cv2
import numpy as np


def fliplr_joints(
    keypoints: np.ndarray, img_width: int, flip_pairs: List[Tuple[int, int]]
) -> np.ndarray:
    """Flip human joints horizontally.

    Args:
        keypoints: Keyponts pair, in shape [N, (x, y)]
        img_width: Image width.
        flip_pairs: Pairs of keypoints which are mirrored
            (for example, left ear -- right ear).

    Returns:
        Flipped human joints.
    """
    assert img_width > 0

    keypoints_flipped = keypoints.copy()

    # Swap left-right parts
    for left, right in flip_pairs:
        keypoints_flipped[left, :] = keypoints[right, :]
        keypoints_flipped[right, :] = keypoints[left, :]

    # Flip horizontally
    # keypoints is usually stored as integer from 0 to image_size - 1
    keypoints_flipped[:, 0] = img_width - 1 - keypoints_flipped[:, 0]

    return keypoints_flipped


def get_affine_transform(
    center: np.ndarray,
    scale: np.ndarray,
    rot: float,
    output_size: Tuple[int, int],
    shift: Tuple[float, float] = (0.0, 0.0),
    inv: bool = False,
    pixel_std: float = 200.0,
) -> np.ndarray:
    """Get the affine transform matrix, given the center/scale/rot/output_size.

    Args:
        center: Center of the bounding box (x, y).
        scale: Scale of the bounding box with respect to [width, height].
        rot: Rotation angle in degree.
        output_size: Size of the destination heatmaps, in height, width.
        shift: Shift translation ratio wrt the width/height. Default: [0., 0.].
        inv: Whether to nverse the affine transform direction. Default: False
        pixel_std: The scaling factor. Default: 200.

    Returns:
        The transform matrix.
    """
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2
    assert len(shift) == 2

    scale_tmp = scale * pixel_std

    shift = np.array(shift)
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = rotate_point([0.0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0.0, dst_w * -0.5])

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt: Tuple[float, float], trans_mat: np.ndarray) -> np.ndarray:
    """Apply an affine transformation to the points.

    Args:
        pt: a 2 dimensional point to be transformed
        trans_mat: 2x3 matrix of an affine transform

    Returns:
        Transformed points.
    """
    assert len(pt) == 2
    new_pt = np.array(trans_mat) @ np.array([pt[0], pt[1], 1.0])

    return new_pt


def rotate_point(pt: Tuple[float, float], angle_rad: float) -> Tuple[float, float]:
    """Rotate a point by an angle.

    Args:
        pt: 2 dimensional point to be rotated
        angle_rad: rotation angle by radian

    Returns:
        Rotated point.
    """
    assert len(pt) == 2
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    new_x = pt[0] * cs - pt[1] * sn
    new_y = pt[0] * sn + pt[1] * cs
    rotated_pt = [new_x, new_y]

    return rotated_pt


def _get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """To calculate the affine matrix, three pairs of points are required. This
    function is used to get the 3rd point, given 2D points a & b.

    The 3rd point is defined by rotating vector `a - b` by 90 degrees
    anticlockwise, using b as the rotation center.

    Args:
        a: point(x,y)
        b: point(x,y)

    Returns:
        The 3rd point.
    """
    assert len(a) == 2
    assert len(b) == 2
    direction = a - b
    third_pt = b + np.array([-direction[1], direction[0]], dtype=np.float32)

    return third_pt


def get_warp_matrix(
    theta: float, size_input: np.ndarray, size_dst: np.ndarray, size_target: np.ndarray
) -> np.ndarray:
    """Calculate the transformation matrix based on Unbiased Data Processing (UDP)

    Args:
        theta: Rotation angle in degrees.
        size_input: Size of input image [w, h].
        size_dst: Size of output image [w, h].
        size_target: Size of ROI in input plane [w, h].

    Returns:
        A matrix for transformation.
    """
    theta = np.deg2rad(theta)
    matrix = np.zeros((2, 3), dtype=np.float32)
    scale_x = size_dst[0] / size_target[0]
    scale_y = size_dst[1] / size_target[1]
    matrix[0, 0] = np.cos(theta) * scale_x
    matrix[0, 1] = -np.sin(theta) * scale_x
    matrix[0, 2] = scale_x * (
        -0.5 * size_input[0] * np.cos(theta)
        + 0.5 * size_input[1] * np.sin(theta)
        + 0.5 * size_target[0]
    )
    matrix[1, 0] = np.sin(theta) * scale_y
    matrix[1, 1] = np.cos(theta) * scale_y
    matrix[1, 2] = scale_y * (
        -0.5 * size_input[0] * np.sin(theta)
        - 0.5 * size_input[1] * np.cos(theta)
        + 0.5 * size_target[1]
    )
    return matrix


def warp_affine_joints(joints: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """Apply affine transformation defined by the transform matrix on the joints.

    Args:
        joints: Origin coordinate of joints.
        mat: The affine matrix.

    Returns:
        Result coordinate of joints.

    """
    joints = np.array(joints)
    shape = joints.shape
    joints = joints.reshape(-1, 2)
    warped_joints = np.dot(
        np.concatenate((joints, joints[:, 0:1] * 0 + 1), axis=1), mat.T
    ).reshape(shape)
    return warped_joints

from collections import defaultdict
from typing import List

import numpy as np
import scipy.optimize


def _max_match(scores: np.ndarray) -> np.ndarray:
    assoc = scipy.optimize.linear_sum_assignment(scores)
    assoc = np.array(assoc).T.astype(np.int32)
    return assoc


def match_by_tag(
    val_k: np.ndarray,
    tag_k: np.ndarray,
    ind_k: np.ndarray,
    joint_order: List[int],
    vis_thr: float = 0.1,
    tag_thr: float = 1,
    ignore_too_much: bool = False,
    use_rounded_norm: bool = True,
) -> np.ndarray:
    """Perform the matching used in associative embedding

    Args:
        val_k: heatmap value for top M. In shape [K, M]
        tag_k: tag value for top M. In shape [K, M, L]
        ind_k: heatmap location for top M. In shape [K, M, 2(x, y)]
        joint_order: The order to perform grouping
        vis_thr: The heatmap threshold. Default: 0.1
        tag_thr: The tag threshold. Default: 1.
        ignore_too_much: Drop the matching if the number of instance is larger than M.
            Default: False
        use_rounded_norm: Use rounded norm during grouping. Default: True

    Returns:
        The matched result, in shape [M, K, 4]. Return empty array if none is matched.

    """
    # tag_k: K, M, L
    num_joints, max_num, _ = tag_k.shape

    default_ = np.zeros((num_joints, 3 + tag_k.shape[2]), np.float32)
    joint_k = np.concatenate((ind_k, val_k[..., None], tag_k), axis=2)

    joint_dict = defaultdict(lambda: default_.copy())
    tag_dict = dict()

    for i in range(num_joints):
        idx = joint_order[i]

        tags = tag_k[idx]
        joints = joint_k[idx]
        mask = joints[:, 2] > vis_thr

        tags = tags[mask]
        if tags.shape[0] == 0:
            continue

        joints = joints[mask]

        if i == 0 or len(joint_dict) == 0:
            for j in range(tags.shape[0]):
                key = tags[j, 0]
                joint_dict[key][idx] = joints[j]
                tag_dict[key] = [tags[j]]
        else:
            grouped_keys = list(joint_dict.keys())

            grouped_tags = [
                np.mean(np.stack(tag_dict[x]), axis=0) for x in grouped_keys
            ]

            if ignore_too_much and len(grouped_keys) == max_num:
                continue

            grouped_tags = np.stack(grouped_tags)

            diff = joints[:, None, 3:] - grouped_tags[None, :, :]
            diff_normed = np.linalg.norm(diff, ord=2, axis=2)
            diff_saved = diff_normed.copy()

            if use_rounded_norm:
                diff_normed = np.round(diff_normed)

            num_added = diff.shape[0]
            num_grouped = diff.shape[1]

            if num_added > num_grouped:
                diff_normed = np.concatenate(
                    (
                        diff_normed,
                        np.zeros((num_added, num_added - num_grouped), np.float32)
                        + 1e10,
                    ),
                    axis=1,
                )

            pairs = _max_match(diff_normed)
            for row, col in pairs:
                if (
                    row < num_added
                    and col < num_grouped
                    and diff_saved[row][col] < tag_thr
                ):
                    key = grouped_keys[col]
                    joint_dict[key][idx] = joints[row]
                    tag_dict[key].append(tags[row])
                else:
                    key = tags[row, 0]
                    joint_dict[key][idx] = joints[row]
                    tag_dict[key] = [tags[row]]

    ans = np.array(list(joint_dict.values())).astype(np.float32)
    return ans

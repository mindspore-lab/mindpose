#!/usr/bin/env python
"""Visualize the heatmap of the ground truth on the augmented training images
"""
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../..")))

import logging
from argparse import Namespace

import cv2
import mindspore as ms
import numpy as np
from common.config import parse_args
from common.log import setup_default_logging
from mindpose.data import create_dataset, create_pipeline

ms.set_seed(1)

_logger = logging.getLogger(__name__)


def visual_gt_heatmap(args: Namespace) -> None:
    # create dataset
    train_dataset = create_dataset(
        args.train_root,
        args.train_label,
        dataset_format=args.dataset_format,
        is_train=True,
        num_joints=args.num_joints,
        num_workers=args.num_parallel_workers,
        config=args.dataset_setting,
    )

    # create pipeline
    train_dataset = create_pipeline(
        train_dataset,
        transforms=args.train_transforms,
        method=args.pipeline_method,
        batch_size=args.batch_size,
        is_train=True,
        normalize=False,
        hwc_to_chw=False,
        num_workers=args.num_parallel_workers,
        config=args.dataset_setting,
    )

    for i, data in enumerate(
        train_dataset.create_dict_iterator(num_epochs=1, output_numpy=True)
    ):
        # visualize the first 10 images only
        if i > 10:
            break

        img = data["image"][0][..., ::-1]  # RGB to BGR
        heatmap = data["target"][0] * 255

        # in case of muli-resolution heatmap, choose the largest one
        if len(heatmap.shape) == 4:
            heatmap = heatmap[-1]
        K, H, W = heatmap.shape
        heatmap = np.sum(heatmap, axis=0)
        heatmap = np.clip(heatmap, 0, 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.resize(
            heatmap, dsize=(img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR
        )

        img = img * 0.7 + heatmap * 0.3

        if not os.path.isdir(args.outdir):
            os.makedirs(args.outdir)

        fpath = os.path.join(args.outdir, f"{i}_gt.jpg")
        _logger.info(f"Saving to {fpath}")
        cv2.imwrite(fpath, img)

        if "mask" in data:
            mask = data["mask"][0] * 255
            if len(mask.shape) == 3:
                mask = mask[-1]
            mask = cv2.resize(
                mask,
                dsize=(img.shape[1], img.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            fpath = os.path.join(args.outdir, f"{i}_mask.jpg")
            _logger.info(f"Saving to {fpath}")
            cv2.imwrite(fpath, mask)

        if "tag_ind" in data:
            tag_ind = data["tag_ind"][0]  # S, M, K, 2
            if len(tag_ind.shape) == 4:
                tag_ind = tag_ind[-1]
            M = tag_ind.shape[0]
            tag_mask = np.zeros((M, K, H * W), dtype=np.uint8)
            np.put_along_axis(tag_mask, tag_ind[..., 0:1], tag_ind[..., 1:2], axis=2)
            tag_mask = tag_mask.reshape(M, K, H, W)
            tag_mask = tag_mask.sum(axis=(0, 1))
            tag_mask *= 255
            tag_mask = cv2.resize(
                tag_mask,
                dsize=(img.shape[1], img.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            fpath = os.path.join(args.outdir, f"{i}_tag_mask.jpg")
            _logger.info(f"Saving to {fpath}")
            cv2.imwrite(fpath, tag_mask)


def main():
    setup_default_logging()

    args = parse_args(
        description="Visualize the heatmap of the ground truth "
        "on the augmented training images."
    )
    visual_gt_heatmap(args)


if __name__ == "__main__":
    main()

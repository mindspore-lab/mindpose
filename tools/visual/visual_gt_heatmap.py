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
from mindpose.data import create_dataset, create_pipeline

ms.set_seed(1)


def visual_gt_heatmap(args: Namespace) -> None:
    # create dataset
    train_dataset = create_dataset(
        args.train_root,
        args.train_label,
        dataset_format=args.dataset_format,
        is_train=True,
        num_workers=args.num_parallel_workers,
        config=args.dataset_detail,
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
        config=args.dataset_detail,
    )

    for i, data in enumerate(
        train_dataset.create_dict_iterator(num_epochs=1, output_numpy=True)
    ):
        # visualize the first 10 images only
        if i > 10:
            break

        img = data["image"][0][..., ::-1]  # RGB to BGR
        heatmap = data["target"][0] * 255
        heatmap = np.sum(heatmap, axis=0)
        heatmap = np.clip(heatmap, 0, 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, dsize=(img.shape[1], img.shape[0]))

        img = img * 0.7 + heatmap * 0.3

        if not os.path.isdir(args.outdir):
            os.makedirs(args.outdir)

        fpath = os.path.join(args.outdir, f"{i}_gt.jpg")
        logging.info(f"Saving to {fpath}")
        cv2.imwrite(fpath, img)


def main():
    args = parse_args(
        description="Visualize the heatmap of the ground truth "
        "on the augmented training images."
    )
    visual_gt_heatmap(args)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    main()

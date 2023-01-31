#!/usr/bin/env python
"""Visualize the heatmap of the prediction on the cropped validation images
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
from mindpose.models import create_decoder, create_eval_network, create_network


def visual_pred_heatmap(args: Namespace) -> None:
    # create dataset
    dataset = create_dataset(
        args.val_root,
        args.val_label,
        dataset_format=args.dataset_format,
        is_train=False,
        use_gt_bbox_for_val=args.val_use_gt_bbox,
        detection_file=args.val_detection_result,
        num_workers=args.num_parallel_workers,
        config=args.dataset_detail,
    )

    # create pipeline
    dataset = create_pipeline(
        dataset,
        transforms=args.val_transforms,
        method=args.pipeline_method,
        batch_size=args.batch_size,
        is_train=False,
        normalize_mean=args.normalize_mean,
        normalize_std=args.normalize_std,
        config=args.dataset_detail,
    )

    # create network
    net = create_network(
        args.backbone_name,
        args.head_name,
        neck_name=args.neck_name,
        in_channels=args.in_channels,
        neck_out_channels=args.neck_out_channels,
        num_joints=args.num_joints,
    )
    ms.load_checkpoint(args.ckpt, net, strict_load=False)

    # create evaluation network
    decoder = create_decoder(args.decoder_name, to_original=False)
    net = create_eval_network(net, decoder, output_raw=True)

    for i, data in enumerate(dataset.create_dict_iterator(num_epochs=1)):
        # visualize the first 10 images only
        if i > 10:
            break

        (keypoint, _), heatmap = net(
            data["image"], data["center"], data["scale"], data["bbox_scores"]
        )

        img = data["image"].asnumpy()[0]
        std = np.array(args.normalize_std)
        mean = np.array(args.normalize_mean)
        img = img * std[:, None, None] + mean[:, None, None]
        img = img * 255.0
        img = np.array(img.round(), dtype=np.uint8)
        img = np.transpose(img, (1, 2, 0))
        img = np.clip(img, 0, 255)
        img = img[..., ::-1].astype(np.uint8).copy()

        mask = np.zeros(heatmap.shape[2:], dtype=np.uint8)
        keypoint = keypoint.asnumpy()[0]
        for j in range(keypoint.shape[0]):
            cv2.circle(
                mask, (int(keypoint[j][0]), int(keypoint[j][1])), 0, [255, 255, 255], -1
            )
        mask = cv2.resize(
            mask, dsize=(img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST
        )

        heatmap = heatmap.asnumpy()[0] * 255
        heatmap = np.sum(heatmap, axis=0)
        heatmap = np.clip(heatmap, 0, 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, dsize=(img.shape[1], img.shape[0]))

        img[mask > 0] = np.array([0, 0, 255], dtype=np.uint8)
        img = img * 0.7 + heatmap * 0.3

        if not os.path.isdir(args.outdir):
            os.makedirs(args.outdir)

        fpath = os.path.join(args.outdir, f"{i}_pred.jpg")
        logging.info(f"Saving to {fpath}")
        cv2.imwrite(fpath, img)


def main():
    args = parse_args(
        description="Visualize the heatmap of the prediction on the cropped validation images",
        need_ckpt=True,
    )
    visual_pred_heatmap(args)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    main()

#!/usr/bin/env python
"""Visualize the prediciton of keypoint on the validation images
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
from common.config import parse_args
from common.log import setup_default_logging
from mindpose.data import create_dataset, create_pipeline
from mindpose.models import create_decoder, create_eval_network, create_network

_logger = logging.getLogger(__name__)


def visual_pred_keypoint(args: Namespace) -> None:
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
        num_workers=args.num_parallel_workers,
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
    decoder = create_decoder(args.decoder_name, **args.decoder_detail)
    net = create_eval_network(net, decoder, output_raw=False)

    for i, data in enumerate(dataset.create_dict_iterator(num_epochs=1)):
        if i > 10:
            break

        keypoint, _ = net(
            data["image"], data["center"], data["scale"], data["bbox_scores"]
        )

        image_path = data["image_file"].asnumpy()[0]
        img = cv2.imread(image_path)

        keypoint = keypoint.asnumpy()[0]
        for j in range(keypoint.shape[0]):
            cv2.circle(
                img, (int(keypoint[j][0]), int(keypoint[j][1])), 2, [0, 0, 255], -1
            )

        if not os.path.isdir(args.outdir):
            os.makedirs(args.outdir)

        fpath = os.path.join(args.outdir, f"{i}_pred.jpg")
        _logger.info(f"Saving to {fpath}")
        cv2.imwrite(fpath, img)


def main():
    setup_default_logging()

    args = parse_args(
        description="Visualize the prediciton of keypoint on the validation images",
        need_ckpt=True,
    )
    visual_pred_keypoint(args)


if __name__ == "__main__":
    main()

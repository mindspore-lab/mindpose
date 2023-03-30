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
from common.log import setup_default_logging
from mindpose.data import create_dataset, create_pipeline
from mindpose.models import create_decoder, create_eval_network, create_network
from mindpose.utils.match import match_by_tag

_logger = logging.getLogger(__name__)


def visual_pred_heatmap(args: Namespace) -> None:
    # create dataset
    dataset = create_dataset(
        args.val_root,
        args.val_label,
        dataset_format=args.dataset_format,
        is_train=False,
        num_joints=args.num_joints,
        use_gt_bbox_for_val=args.val_use_gt_bbox,
        detection_file=args.val_detection_result,
        num_workers=args.num_parallel_workers,
        config=args.dataset_setting,
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
        config=args.dataset_setting,
    )

    # create network
    net = create_network(
        args.backbone_name,
        args.head_name,
        neck_name=args.neck_name,
        backbone_pretrained=False,
        in_channels=args.in_channels,
        neck_out_channels=args.neck_out_channels,
        num_joints=args.num_joints,
        backbone_args=args.backbone_setting,
        neck_args=args.neck_setting,
        head_args=args.head_setting,
    )
    ms.load_checkpoint(args.ckpt, net, strict_load=False)

    # create evaluation network
    decoder = create_decoder(args.decoder_name, **args.decoder_setting)
    net = create_eval_network(net, decoder, output_raw=False)

    for i, data in enumerate(dataset.create_dict_iterator(num_epochs=1)):
        # visualize the first 10 images only
        if i > 10:
            break

        val_k, tag_k, ind_k, heatmap, tag_heatmap = net(data["image"], data["mask"])

        img = data["image"].asnumpy()[0]
        std = np.array(args.normalize_std)
        mean = np.array(args.normalize_mean)
        img = img * std[:, None, None] + mean[:, None, None]
        img = img * 255.0
        img = np.array(img.round(), dtype=np.uint8)
        img = np.transpose(img, (1, 2, 0))
        img = np.clip(img, 0, 255)
        img = img[..., ::-1].astype(np.uint8).copy()

        tag_mask = np.zeros(heatmap.shape[2:], dtype=np.uint8)
        group_mask = np.zeros(heatmap.shape[2:], dtype=np.uint8)
        ind_k = ind_k.asnumpy()[0]  # K x M x 2
        val_k = val_k.asnumpy()[0]  # K x M
        tag_k = tag_k.asnumpy()[0]  # K x M x 1
        grouped = match_by_tag(val_k, tag_k, ind_k, args.eval_setting["joint_order"])
        tag_k = (tag_k - tag_k.min()) / (tag_k.max() - tag_k.min()) + 0.2
        tag_k /= 1.2
        for j in range(ind_k.shape[0]):
            for k in range(ind_k.shape[1]):
                if val_k[j, k] > args.eval_setting["vis_thr"]:
                    tag_v = int((tag_k[j, k, 0] * 255).clip(0, 255))
                    cv2.circle(
                        tag_mask,
                        (int(ind_k[j, k, 0]), int(ind_k[j, k, 1])),
                        0,
                        tag_v,
                        -1,
                    )

        for j, single in enumerate(grouped):
            v = int((j + 1) / (len(grouped) + 1) * 255)
            for kp in single:
                if kp[2] > 0:
                    cv2.circle(group_mask, (int(kp[0]), int(kp[1])), 0, v, -1)
        tag_mask = cv2.resize(
            tag_mask,
            dsize=(img.shape[1], img.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        group_mask = cv2.resize(
            group_mask,
            dsize=(img.shape[1], img.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        tag_mask = cv2.applyColorMap(tag_mask, cv2.COLORMAP_JET)
        group_mask = cv2.applyColorMap(group_mask, cv2.COLORMAP_JET)

        heatmap = heatmap.asnumpy()[0] * 255
        heatmap = np.sum(heatmap, axis=0)
        heatmap = np.clip(heatmap, 0, 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, dsize=(img.shape[1], img.shape[0]))

        tag_heatmap = tag_heatmap.asnumpy()[0]
        tag_heatmap = (tag_heatmap - tag_heatmap.min(axis=(1, 2), keepdims=True)) / (
            tag_heatmap.max(axis=(1, 2), keepdims=True)
            - tag_heatmap.min(axis=(1, 2), keepdims=True)
        )
        tag_heatmap = tag_heatmap.mean(axis=0)
        tag_heatmap *= 255
        tag_heatmap = np.clip(tag_heatmap, 0, 255).astype(np.uint8)
        tag_heatmap = cv2.applyColorMap(tag_heatmap, cv2.COLORMAP_JET)
        tag_heatmap = cv2.resize(tag_heatmap, dsize=(img.shape[1], img.shape[0]))

        if not os.path.isdir(args.outdir):
            os.makedirs(args.outdir)

        img_heatmap = img * 0.7 + heatmap * 0.3

        fpath = os.path.join(args.outdir, f"{i}_heatmap.jpg")
        _logger.info(f"Saving to {fpath}")
        cv2.imwrite(fpath, img_heatmap)

        img_tag = img * 0.3 + tag_heatmap * 0.7

        fpath = os.path.join(args.outdir, f"{i}_tag_heatmap.jpg")
        _logger.info(f"Saving to {fpath}")
        cv2.imwrite(fpath, img_tag)

        img_embedding = img * 0.3 + tag_mask * 0.7

        fpath = os.path.join(args.outdir, f"{i}_embedding.jpg")
        _logger.info(f"Saving to {fpath}")
        cv2.imwrite(fpath, img_embedding)

        img_group = img * 0.3 + group_mask * 0.7

        fpath = os.path.join(args.outdir, f"{i}_group.jpg")
        _logger.info(f"Saving to {fpath}")
        cv2.imwrite(fpath, img_group)


def main():
    setup_default_logging()

    args = parse_args(
        description="Visualize the heatmap of the prediction on "
        "the cropped validation images",
        need_ckpt=True,
    )
    visual_pred_heatmap(args)


if __name__ == "__main__":
    main()

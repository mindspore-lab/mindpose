#!/usr/bin/env python
"""Perform evaluation on the validation dataset
"""
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

import json
import logging
from argparse import Namespace

import mindspore as ms
from common.config import parse_args
from common.log import setup_default_logging
from mindpose.data import create_dataset, create_pipeline
from mindpose.engine import create_evaluator, create_inferencer
from mindpose.models import create_decoder, create_eval_network, create_network

_logger = logging.getLogger(__name__)


def eval(args: Namespace) -> None:
    # set up mindspore running mode
    ms.set_context(mode=args.mode)

    # create validation dataset
    val_dataset = create_dataset(
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

    val_dataset = create_pipeline(
        val_dataset,
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

    # add decoder head
    decoder = create_decoder(args.decoder_name, **args.decoder_setting)
    net = create_eval_network(net, decoder)

    # create inferencer
    inferencer = create_inferencer(
        net=net,
        name=args.inference_method,
        config=args.eval_setting,
        dataset_config=args.dataset_setting,
        decoder=decoder,
        progress_bar=True,
    )

    # create evaluator
    os.makedirs(args.outdir, exist_ok=True)
    keypoint_result_tmp_path = os.path.join(args.outdir, "result_keypoint.json")
    evaluator = create_evaluator(
        annotation_file=args.val_label,
        name=args.eval_method,
        metric=args.eval_metric,
        config=args.eval_setting,
        dataset_config=args.dataset_setting,
        result_path=keypoint_result_tmp_path,
    )

    # inference on the whole dataset
    outputs = inferencer(dataset=val_dataset)

    # perform evaluation
    result = evaluator(outputs)
    result_path = os.path.join(args.outdir, "result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=4)
    _logger.info(result)
    _logger.info(f"Result is saved at `{result_path}`.")


def main():
    setup_default_logging()
    args = parse_args(description="Evaluation script", need_ckpt=True)
    eval(args)


if __name__ == "__main__":
    main()

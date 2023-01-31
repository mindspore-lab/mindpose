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
from mindpose.data import create_dataset, create_pipeline
from mindpose.engine.evaluators import create_evaluator
from mindpose.engine.inferencer import create_inferencer
from mindpose.models import create_decoder, create_eval_network, create_network


def eval(args: Namespace) -> None:
    # set up mindspore running mode
    ms.set_context(mode=args.mode)

    # create validation dataset
    val_dataset = create_dataset(
        args.val_root,
        args.val_label,
        dataset_format=args.dataset_format,
        is_train=False,
        use_gt_bbox_for_val=args.val_use_gt_bbox,
        detection_file=args.val_detection_result,
        num_workers=args.num_parallel_workers,
        config=args.dataset_detail,
    )

    val_dataset = create_pipeline(
        val_dataset,
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

    # add decoder head
    decoder = create_decoder(args.decoder_name)
    net = create_eval_network(net, decoder)

    # create inferencer
    inferencer = create_inferencer(
        net,
        config=args.eval_args,
        name=args.inference_method,
        decoder=decoder,
        progress_bar=True,
    )

    # create evaluator
    os.makedirs(args.outdir, exist_ok=True)
    keypoint_result_tmp_path = os.path.join(args.outdir, "result_keypoint.json")
    evaluator = create_evaluator(
        args.eval_method,
        config=args.eval_args,
        annotation_file=args.val_label,
        metric=args.metric,
        result_path=keypoint_result_tmp_path,
    )

    # inference on the whole dataset
    outputs = inferencer(dataset=val_dataset)

    # perform evaluation
    result = evaluator.evaluate(outputs)
    with open(os.path.join(args.outdir, "result.json"), "w") as f:
        json.dump(result, f, indent=4)


def main():
    args = parse_args(description="Evaluation script", need_ckpt=True)
    eval(args)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    main()

#!/usr/bin/env python
"""Perform training
"""
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

import logging
from argparse import Namespace

import mindspore as ms
from common.config import parse_args
from mindpose.callbacks import EvalCallback
from mindpose.data import create_dataset, create_pipeline
from mindpose.engine import create_evaluator, create_inferencer
from mindpose.models import (
    create_decoder,
    create_eval_network,
    create_loss,
    create_network,
    create_network_with_loss,
)
from mindpose.utils.initializer import init_by_kaiming_uniform
from mindpose.utils.lr_scheduler import create_lr_scheduler
from mindpose.utils.optimizer import create_optimizer
from mindspore import FixedLossScaleManager, Model

ms.set_seed(0)


def train(args: Namespace) -> None:
    # set up mindspore running mode
    ms.set_context(mode=args.mode)
    if args.mode == 0:
        ms.set_context(enable_graph_kernel=args.enable_graph_kernel)

    # set up distribution mode
    if args.distribute:
        ms.communication.init()
        device_num = ms.communication.get_group_size()
        rank_id = ms.communication.get_rank()
        ms.set_auto_parallel_context(
            device_num=device_num,
            parallel_mode="data_parallel",
            gradients_mean=True,
            parameter_broadcast=True,
        )

        if "DEVICE_ID" in os.environ:
            ms.set_context(device_id=int(os.environ["DEVICE_ID"]))
    else:
        device_num = None
        rank_id = None

    # create dataset
    train_dataset = create_dataset(
        args.train_root,
        args.train_label,
        dataset_format=args.dataset_format,
        is_train=True,
        device_num=device_num,
        rank_id=rank_id,
        num_workers=args.num_parallel_workers,
        config=args.dataset_detail,
    )

    val_dataset = create_dataset(
        args.val_root,
        args.val_label,
        dataset_format=args.dataset_format,
        is_train=False,
        device_num=device_num,
        rank_id=rank_id,
        use_gt_bbox_for_val=args.val_use_gt_bbox,
        detection_file=args.val_detection_result,
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
        normalize_mean=args.normalize_mean,
        normalize_std=args.normalize_std,
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
        num_workers=args.num_parallel_workers,
        config=args.dataset_detail,
    )

    # create network
    net = create_network(
        args.backbone_name,
        args.head_name,
        neck_name=args.neck_name,
        backbone_pretrained=args.backbone_pretrained,
        backbone_ckpt_url=args.backbone_ckpt_url,
        in_channels=args.in_channels,
        neck_out_channels=args.neck_out_channels,
        num_joints=args.num_joints,
    )
    if args.init_by_kaiming_uniform:
        init_by_kaiming_uniform(net)

    # create evaluation network
    decoder = create_decoder(args.decoder_name, **args.decoder_detail)
    val_net = create_eval_network(net, decoder)

    # create loss
    loss = create_loss(args.loss, **args.loss_detail)

    # create net_with_loss
    net_with_loss = create_network_with_loss(
        net, loss, has_extra_inputs=args.loss_with_extra_input
    )

    # rescale the learning rate
    if args.distribute and args.scale_lr:
        lr = args.lr * device_num
        logging.info(
            "Rescale the learning rate by linear rule. " f"New learning rate = {lr}"
        )
    else:
        lr = args.lr

    # create learning rate scheduler
    lr_scheduler = create_lr_scheduler(
        name=args.scheduler,
        max_lr=lr,
        total_epochs=args.num_epochs,
        steps_per_epoch=train_dataset.get_dataset_size(),
        warmup=args.warmup,
        **args.scheduler_detail,
    )

    # create optimizer
    optimizer = create_optimizer(
        net_with_loss.trainable_params(),
        name=args.optimizer,
        learning_rate=lr_scheduler,
        loss_scale=args.loss_scale,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
    )

    # create model
    loss_scale_manager = FixedLossScaleManager(
        loss_scale=args.loss_scale, drop_overflow_update=False
    )
    model = Model(
        network=net_with_loss,
        optimizer=optimizer,
        amp_level=args.amp_level,
        loss_scale_manager=loss_scale_manager,
    )

    # create inferencer and evaluator
    inferencer = create_inferencer(
        net=val_net,
        name=args.inference_method,
        config=args.eval_detail,
        dataset_config=args.dataset_detail,
        decoder=decoder,
    )

    keypoint_result_tmp_path = os.path.join(args.outdir, "result_keypoint.json")
    evaluator = create_evaluator(
        annotation_file=args.val_label,
        name=args.eval_method,
        metric=args.eval_metric,
        config=args.eval_detail,
        dataset_config=args.dataset_detail,
        result_path=keypoint_result_tmp_path,
    )

    # create callbacks for loss monitor and perform evaluation
    model_outdir = os.path.join(args.outdir, "saved_model")
    os.makedirs(model_outdir, exist_ok=True)
    summary_outdir = os.path.join(args.outdir, "summary")
    best_ckpt_path = os.path.join(model_outdir, "hrnet_best.ckpt")
    last_ckpt_path = os.path.join(model_outdir, "hrnet_last.ckpt")
    eval_cb = EvalCallback(
        inferencer,
        evaluator,
        val_dataset,
        interval=args.val_interval,
        max_epoch=args.num_epochs,
        save_best=True,
        save_last=True,
        net_to_save=net_with_loss,
        best_ckpt_path=best_ckpt_path,
        last_ckpt_path=last_ckpt_path,
        summary_dir=summary_outdir,
        rank_id=rank_id,
        device_num=device_num,
    )
    cb = [eval_cb]

    # start training
    model.train(args.num_epochs, train_dataset, callbacks=cb, dataset_sink_mode=True)


def main():
    args = parse_args(description="Training script")
    train(args)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    main()

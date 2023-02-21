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
from common.log import setup_default_logging
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
from mindpose.optim import create_optimizer
from mindpose.scheduler import create_lr_scheduler
from mindspore import DynamicLossScaleManager, Model

ms.set_seed(0)

_logger = logging.getLogger(__name__)


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
            device_num=device_num, parallel_mode="data_parallel", gradients_mean=True
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
        config=args.dataset_setting,
    )

    val_dataset = create_dataset(
        args.val_root,
        args.val_label,
        dataset_format=args.dataset_format,
        is_train=False,
        use_gt_bbox_for_val=args.val_use_gt_bbox,
        detection_file=args.val_detection_result,
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
        normalize_mean=args.normalize_mean,
        normalize_std=args.normalize_std,
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
        backbone_pretrained=args.backbone_pretrained,
        backbone_ckpt_url=args.backbone_ckpt_url,
        in_channels=args.in_channels,
        neck_out_channels=args.neck_out_channels,
        num_joints=args.num_joints,
        backbone_args=args.backbone_setting,
        neck_args=args.neck_setting,
        head_args=args.head_setting,
    )
    num_params = sum([param.size for param in net.get_parameters()])
    _logger.info(f"Model param: {num_params}")

    # create evaluation network
    decoder = create_decoder(args.decoder_name, **args.decoder_setting)
    val_net = create_eval_network(net, decoder)

    # create loss
    loss = create_loss(args.loss, **args.loss_setting)

    # create net_with_loss
    net_with_loss = create_network_with_loss(
        net, loss, has_extra_inputs=args.loss_with_extra_input
    )

    # create learning rate scheduler
    lr_scheduler = create_lr_scheduler(
        name=args.scheduler,
        lr=args.lr,
        total_epochs=args.num_epochs,
        steps_per_epoch=train_dataset.get_dataset_size(),
        warmup=args.warmup,
        **args.lr_scheduler_setting,
    )

    # create optimizer
    optimizer = create_optimizer(
        net_with_loss.trainable_params(),
        name=args.optimizer,
        learning_rate=lr_scheduler,
        filter_bias_and_bn=args.filter_bias_and_bn,
        weight_decay=args.weight_decay,
        **args.optimizer_setting,
    )

    # load the checkpoint if provided
    if args.ckpt:
        _logger.info(f"Loading the checkpoint from {args.ckpt}")
        param_dict = ms.load_checkpoint(args.ckpt)
        ms.load_param_into_net(net_with_loss, param_dict)
        ms.load_param_into_net(optimizer, param_dict)

    # create model
    loss_scale_manager = DynamicLossScaleManager()
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
        config=args.eval_setting,
        dataset_config=args.dataset_setting,
        decoder=decoder,
    )

    keypoint_result_tmp_path = os.path.join(args.outdir, "result_keypoint.json")
    evaluator = create_evaluator(
        annotation_file=args.val_label,
        name=args.eval_method,
        metric=args.eval_metric,
        config=args.eval_setting,
        dataset_config=args.dataset_setting,
        result_path=keypoint_result_tmp_path,
    )

    # create callbacks for loss monitor and perform evaluation
    model_outdir = os.path.join(args.outdir, "saved_model")
    os.makedirs(model_outdir, exist_ok=True)
    summary_outdir = os.path.join(args.outdir, "summary")
    # determine the model name
    model_name = os.path.basename(args.config).replace(".yaml", "")
    best_ckpt_path = os.path.join(model_outdir, f"{model_name}_best.ckpt")
    last_ckpt_path = os.path.join(model_outdir, f"{model_name}_last.ckpt")
    eval_cb = EvalCallback(
        inferencer,
        evaluator,
        val_dataset,
        interval=args.val_interval,
        max_epoch=args.num_epochs,
        save_best=args.save_best,
        save_last=args.save_last,
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
    setup_default_logging()
    args = parse_args(description="Training script")
    train(args)


if __name__ == "__main__":
    main()

# SimpleBaseline

> [Simple Baselines for Human Pose Estimation and Tracking](https://arxiv.org/abs/1804.06208)

## Introduction

There has been significant progress on pose estimation and increasing interests on pose tracking in recent years. At the same time, the overall algorithm and system complexity increases as well, making the algorithm analysis and comparison more difficult. This work provides simple and effective baseline methods. They are helpful for inspiring and evaluating new ideas for the field. State-of-the-art results are achieved on challenging benchmarks


## Results

Our reproduced model performance on COCO2017-val is reported as follows.

<div align="center">

| Model     | Context  | pretrain | mAP@0.5:0.95 | mAP@0.5 | Params (M) | Recipe    | Download    |
|-----------|----------|----------|--------------|---------|------------|-----------|-------------|
| resnet50  | D910x8-G | False    |              |         |            | [yaml]()  | [weights]() |

</div>

#### Notes
- Context: Training context denoted as {device}x{pieces}-{MS mode}, where mindspore mode can be G - graph mode or F - pynative mode with ms function. For example, D910x8-G is for training on 8 pieces of Ascend 910 NPU using graph mode. 
- mAP@0.5:0.95 and mAP@0.5: Accuracy reported on the validation set of COCO2017. 


## Quick Start

### Preparation

#### Installation
Please refer to the [installation instruction]() in MindPose.

#### Dataset Preparation
Please download the [COCO2017](https://cocodataset.org/#home) dataset for model training and validation.
Top-down approach uses person detection result with 56.4 AP for evaluaton. You can download it from [here](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch).

You need to modify the paths of the data in the configure file.

### Training

* Distributed Training

It is easy to reproduce the reported results with the pre-defined training recipe. For distributed training on multiple Ascend 910 devices, please run

```shell
# distrubted training on multiple GPU/Ascend devices
mpirun -n 8 python tools/train.py --config configs/simple_baseline/resnet50_ascend.yaml
```

> If the script is executed by the root user, the `--allow-run-as-root` parameter must be added to `mpirun`.

Similarly, you can train the model on multiple GPU devices with the above `mpirun` command.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on a CPU/GPU/Ascend device
python tools/train.py --config configs/simple_baseline/resnet50_ascend.yaml
```

### Validation

To validate the accuracy of the trained model, you can use `tools/eval.py` and parse the checkpoint path with `--ckpt`.

```shell
python tools/eval.py --config configs/simple_baseline/resnet50_ascend.yaml --ckpt /path/to/ckpt
```

### Deployment

TODO

## References

[1] B. Xiao, H. Wu, and Y. Wei, “Simple Baselines for Human Pose Estimation and Tracking,” presented at the Proceedings of the European Conference on Computer Vision (ECCV), 2018, pp. 466–481.

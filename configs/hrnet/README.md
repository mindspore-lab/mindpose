# HRNet

> [Deep High-Resolution Representation Learning for Human Pose Estimation](https://arxiv.org/abs/1902.09212)

## Introduction

HRNet maintains high-resolution representations through the whole process. It starts from a high-resolution subnetwork as the first stage, gradually adds high-to-low resolution subnetworks one by one to form more stages, and connects the mutli-resolution subnetworks in parallel. Conducting repeated multi-scale fusions such that each of the high-to-low resolution representations receives information from other parallel representations over and over, leads to rich high-resolution representations. As a result, the predicted keypoint heatmap is potentially more accurate and spatially more precise.


## Results

Our reproduced model performance on COCO2017-val is reported as follows.

<div align="center">

| Model     | Context  | pretrain | mAP@0.5:0.95 | mAP@0.5 | Params (M) | Recipe    | Download    |
|-----------|----------|----------|--------------|---------|------------|-----------|-------------|
| HRNet-w32 | D910x8-G | False    |              |         |            | [yaml]()  | [weights]() |
| HRNet-w48 | D910x8-G | False    |              |         |            | [yaml]()  | [weights]() |

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
mpirun -n 8 python tools/train.py --config configs/hrnet/hrnet_w32_ascend.yaml
```

> If the script is executed by the root user, the `--allow-run-as-root` parameter must be added to `mpirun`.

Similarly, you can train the model on multiple GPU devices with the above `mpirun` command.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on a CPU/GPU/Ascend device
python tools/train.py --config configs/hrnet/hrnet_w32_ascend.yaml
```

### Validation

To validate the accuracy of the trained model, you can use `tools/eval.py` and parse the checkpoint path with `--ckpt`.

```shell
python tools/eval.py --config configs/hrnet/hrnet_w32_ascend.yaml --ckpt /path/to/ckpt
```

### Deployment

TODO

## References

[1] K. Sun, B. Xiao, D. Liu, and J. Wang, “Deep High-Resolution Representation Learning for Human Pose Estimation,” presented at the Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019, pp. 5693–5703. 

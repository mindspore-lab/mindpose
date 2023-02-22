# HRNet

> [Deep High-Resolution Representation Learning for Human Pose Estimation](https://arxiv.org/abs/1902.09212)

## Introduction

HRNet maintains high-resolution representations through the whole process. It starts from a high-resolution subnetwork as the first stage, gradually adds high-to-low resolution subnetworks one by one to form more stages, and connects the mutli-resolution subnetworks in parallel. Conducting repeated multi-scale fusions such that each of the high-to-low resolution representations receives information from other parallel representations over and over, leads to rich high-resolution representations. As a result, the predicted keypoint heatmap is potentially more accurate and spatially more precise.

## Results

Our reproduced model performance on COCO2017-val is reported as follows.

<div align="center">

| Model     | Context  | Input Size | AP    | AP<sup>50</sup> | AP<sup>75</sup> | AR     | AR<sup>50</sup> | Params (M) | Recipe    | Download    |
|-----------|----------|------------|-------|-----------------|-----------------|--------|-----------------|------------|-----------|-------------|
| HRNet-w32 | D910x8-G | 256x192    | 0.749 | 0.905           | 0.822           | 0.802  | 0.941           | 28.59      |[yaml](https://github.com/mindspore-lab/mindpose/blob/master/configs/hrnet/hrnet_w32_ascend.yaml)   | [weights](https://download.mindspore.cn/toolkits/mindpose/hrnet/hrnet_w32_256_192.ckpt) |
| HRNet-w48 | D910x8-G | 256x192    | 0.756 | 0.910           | 0.826           | 0.807  | 0.946           | 63.68      |[yaml](https://github.com/mindspore-lab/mindpose/blob/master/configs/hrnet/hrnet_w48_ascend.yaml)   | [weights](https://download.mindspore.cn/toolkits/mindpose/hrnet/hrnet_w48_256_192.ckpt) |

</div>

#### Notes
- Context: Training context denoted as {device}x{pieces}-{MS mode}, where mindspore mode can be G - graph mode or F - pynative mode with ms function. For example, D910x8-G is for training on 8 pieces of Ascend 910 NPU using graph mode. 
- AP, AP<sup>50</sup>, AP<sup>75</sup>, AR and AR<sup>50</sup>: Accuracy reported on the validation set of COCO2017. 

## Quick Start

### Preparation

#### Installation
Please refer to the [installation instruction](https://github.com/mindspore-lab/mindpose#installation) in MindPose.

#### Dataset Preparation
Please download the [COCO2017](https://cocodataset.org/#home) dataset for model training and validation.
Top-down approach uses person detection result with 56.4 AP for evaluaton. You can download the correspoinding detection result from [here](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch).

After download the dataset and the detection result, you may need to modify the paths listed in the configure file to the corresponding position. Or you can make the download data look like this:

```
${POSE_ROOT}
|-- data
`-- |-- coco
    `-- |-- annotations
        |   |-- person_keypoints_train2017.json
        |   `-- person_keypoints_val2017.json
        |-- person_detection_results
        |   `-- COCO_val2017_detections_AP_H_56_person.json
        `-- images
            |-- train2017
            |   |-- 000000000009.jpg
            |   |-- 000000000025.jpg
            |   |-- 000000000030.jpg
            |   |-- ... 
            `-- val2017
                |-- 000000000139.jpg
                |-- 000000000285.jpg
                |-- 000000000632.jpg
                |-- ... 
```

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
python tools/train.py --config configs/hrnet/hrnet_w32_ascend.yaml --cfg-options distribute=False
```

### Validation

To validate the accuracy of the trained model, you can use `tools/eval.py` and parse the checkpoint path with `--ckpt`. According to [<a href="#references">1</a>], multiple post-processing steps are used, e.g., horizontal flip TTA. You may need to add these arguments in the `--cfg-options`. The command to reproduce the final result is

```shell
python tools/eval.py --config configs/hrnet/hrnet_w32_ascend.yaml --ckpt /path/to/ckpt --cfg-options val_use_gt_bbox=False eval_setting.hflip_tta=True eval_setting.shift_heatmap=True decoder_setting.shift_coordinate=True
```

## References

[1] K. Sun, B. Xiao, D. Liu, and J. Wang, “Deep High-Resolution Representation Learning for Human Pose Estimation.” arXiv, Feb. 25, 2019. doi: 10.48550/arXiv.1902.09212.

# UDP

> [The Devil is in the Details: Delving into Unbiased Data Processing for Human Pose Estimation](https://arxiv.org/abs/1911.07524)

## Introduction

Being a fundamental component in training and inference, data processing has not been systematically considered in human pose estimation community, to the best of our knowledge. This paper focuses on this problem and finds that the devil of human pose estimation evolution is in the biased data processing. Specifically, by investigating the standard data processing in state-of-the-art approaches mainly including coordinate system transformation and keypoint format transformation (i.e., encoding and decoding), it finds that the results obtained by common flipping strategy are unaligned with the original ones in inference. Moreover, there is a statistical error in some keypoint format transformation methods. Two problems couple together, significantly degrade the pose estimation performance and thus lay a trap for the research community. This trap has given bone to many suboptimal remedies, which are always unreported, confusing but influential. By causing failure in reproduction and unfair in comparison, the unreported remedies seriously impedes the technological development. To tackle this dilemma from the source, this paper proposes Unbiased Data Processing (UDP) consist of two technique aspect for the two aforementioned problems respectively (i.e., unbiased coordinate system transformation and unbiased keypoint format transformation). As a model-agnostic approach and a superior solution, UDP successfully pushes the performance boundary of human pose estimation and offers a higher and more reliable baseline for research community.

## Results

Our reproduced model performance on COCO2017-val is reported as follows.

<div align="center">

| Model         | Context  | Input Size | AP    | AP<sup>50</sup> | AP<sup>75</sup> | AR     | AR<sup>50</sup> | Params (M) | Recipe    | Download    |
|---------------|----------|------------|-------|-----------------|-----------------|--------|-----------------|------------|-----------|-------------|
| Resnet50-UDP  | D910x8-G | 256x192    | 0.726 | 0.898           | 0.797           | 0.779  | 0.935           | 34.05      |[yaml](https://github.com/mindspore-lab/mindpose/blob/master/configs/udp/resnet50_w32_udp_ascend.yaml)   | [weights](https://download.mindspore.cn/toolkits/mindpose/udp/resnet50_udp_256_192.ckpt) |
| Resnet101-UDP | D910x8-G | 256x192    | 0.737 | 0.902           | 0.810           | 0.792  | 0.941           | 53.10      |[yaml](https://github.com/mindspore-lab/mindpose/blob/master/configs/udp/resnet101_w32_udp_ascend.yaml)   | [weights](https://download.mindspore.cn/toolkits/mindpose/udp/resnet101_udp_256_192.ckpt) |
| Resnet152-UDP | D910x8-G | 256x192    | 0.744 | 0.904           | 0.818           | 0.798  | 0.942           | 68.79      |[yaml](https://github.com/mindspore-lab/mindpose/blob/master/configs/udp/resnet152_w32_udp_ascend.yaml)   | [weights](https://download.mindspore.cn/toolkits/mindpose/udp/resnet152_udp_256_192.ckpt) |
| HRNet-w32-UDP | D910x8-G | 256x192    | 0.758 | 0.904           | 0.824           | 0.808  | 0.941           | 28.59      |[yaml](https://github.com/mindspore-lab/mindpose/blob/master/configs/udp/hrnet_w32_udp_ascend.yaml)   | [weights](https://download.mindspore.cn/toolkits/mindpose/udp/hrnet_w32_udp_256_192.ckpt) |
| HRNet-w48-UDP | D910x8-G | 256x192    | 0.767 | 0.908           | 0.832           | 0.816  | 0.945           | 63.68      |[yaml](https://github.com/mindspore-lab/mindpose/blob/master/configs/udp/hrnet_w48_udp_ascend.yaml)   | [weights](https://download.mindspore.cn/toolkits/mindpose/udp/hrnet_w48_udp_256_192.ckpt) |

</div>

#### Notes
- Context: Training context denoted as {device}x{pieces}-{MS mode}, where mindspore mode can be G - graph mode or F - pynative mode with ms function. For example, D910x8-G is for training on 8 pieces of Ascend 910 NPU using graph mode.
- AP, AP<sup>50</sup>, AP<sup>75</sup>, AR and AR<sup>50</sup>: Accuracy reported on the validation set of COCO2017.
- Implementation is based on heatmap-based approach.

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
mpirun -n 8 python tools/train.py --config configs/udp/hrnet_w32_udp_ascend.yaml
```

> If the script is executed by the root user, the `--allow-run-as-root` parameter must be added to `mpirun`.

Similarly, you can train the model on multiple GPU devices with the above `mpirun` command.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on a CPU/GPU/Ascend device
python tools/train.py --config configs/udp/hrnet_w32_udp_ascend.yaml --cfg-options distribute=False
```

### Validation

To validate the accuracy of the trained model, you can use `tools/eval.py` and parse the checkpoint path with `--ckpt`. According to [<a href="#references">1</a>], multiple post-processing steps are used, e.g., horizontal flip TTA. You may need to add these arguments in the `--cfg-options`. The command to reproduce the final result is

```shell
python tools/eval.py --config configs/udp/hrnet_w32_udp_ascend.yaml --ckpt /path/to/ckpt --cfg-options val_use_gt_bbox=False eval_setting.hflip_tta=True decoder_setting.dark_udp_refine=True
```

## References

[1] J. Huang, Z. Zhu, F. Guo, G. Huang, and D. Du, “The Devil is in the Details: Delving into Unbiased Data Processing for Human Pose Estimation.” arXiv, Dec. 30, 2020. doi: 10.48550/arXiv.1911.07524.

# MindPose

[Introduction](#introduction) |
[Installation](#installation) |
[Get Started](#get-started) |
[Tutorials](#tutorials) |
[Model List](#model-list) |
[Supported Algorithms](#supported-algorithms) |
[Notes](#notes) 

## Introduction
MindPose is an open-source toolbox for pose estimation based on [MindSpore](https://www.mindspore.cn/en). It collects a series of classic and SoTA vision models, such as HRNet, along with their pre-trained weights and training strategies. 

<details open>
<summary> Major Features </summary>

- **Easy-to-Use.** MindPose decomposes the vision framework into various configurable components. It is easy to customize your data pipeline, models, and learning pipeline with MindPose: 

```python
>>> import mindpose
# create a model
>>> network = mindpose.create_network(backbone_name='resnet50', head_name="simple_baseline_head")
```

- **State-of-The-Art.** MindPose provides various CNN-based and Transformer-based vision models. Their pretrained weights and performance reports are provided to help users select and reuse the right model.

- **Flexibility and efficiency.** MindPose is built on MindSpore which is an efficent DL framework that can be run on different hardware platforms (GPU/CPU/Ascend). It supports both graph mode for high efficiency and pynative mode for flexibility.

</details>

### Benchmark Results

## Installation

### Dependency

- mindspore >= 1.8.1
- numpy >= 1.20.0
- pyyaml >= 5.3
- opencv_python_headless < 4.3
- xtcocotools >= 1.13
- tqdm
- openmpi 4.0.3 (for distributed mode) 

To install the dependency, please run
```shell
pip install -r requirements.txt
```

MindSpore can be easily installed by following the official [instructions](https://www.mindspore.cn/install) where you can select your hardware platform for the best fit. To run in distributed mode, [openmpi](https://www.open-mpi.org/software/ompi/v4.0/) is required to install.

The following instructions assume the desired dependency is fulfilled. 

### Install with PyPI

The released version of MindPose can be installed via `PyPI` as follows:
```shell
pip install mindpose
```

### Install from Source

The latest version of MindPose can be installed as follows:
```shell
pip install git+https://github.com/mindspore-lab/mindpose.git
```

> Notes: MindPose can be installed on Linux and Mac but not on Windows currently.

## Get Started 

### Hands-on Tutorial

TODO

### Training

It is easy to train your model on a standard or customized dataset using `tools/train.py`, where the training strategy is configured with a yaml config file.

- Config and Training Strategy

You can configure your model and other components by writing a yaml config file. Here is an example of training using a preset yaml file.

```shell
mpirun --allow-run-as-root -n 4 python tools/train.py --config configs/hrnet/hrnet_w32_ascend.yaml
```

- Train on OpenI Platform

To run training on the [OpenI](https://openi.pcl.ac.cn/) cloud platform:

1. Create a new training task on the cloud platform.
2. Use `tools/train_on_openi.py` as the starting file.
3. Add run parameter `config` and specify the path to the yaml config file on the website UI interface.
4. Fill in other blanks on the website and launch the training task.

### Validation

To evalute the model performance, please run `tools/eval.py` 

```shell
# validate a trained checkpoint
python tools/eval.py --config=configs/hrnet/hrnet_w32_ascend.yaml --ckpt=/path/to/model.ckpt 
```

## Tutorials

TODO

## Model List

Currently, MindPose supports the model families listed below. More models with pre-trained weights are under development and will be released soon.

<details open>
<summary> Supported models </summary>

* SimpleBaseline - https://arxiv.org/abs/1804.06208
* HRNet - https://arxiv.org/abs/1908.07919

Please see [configs](./configs) for the details about model performance and pretrained weights.

</details>

## Notes
### What is New

TODO

### How to Contribute

We appreciate all kind of contributions including issues and PRs to make MindPose better. 

Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for the contributing guideline. Please follow the [Model Template and Guideline](mindpose/models/model_template.md) for contributing a model that fits the overall interface :)

### License

This project follows the [Apache License 2.0](LICENSE.md) open-source license.

### Acknowledgement

MindPose is an open-source project jointly developed by the MindSpore team.
Sincere thanks to all participating researchers and developers for their hard work on this project.
We also acknowledge the computing resources provided by [OpenI](https://openi.pcl.ac.cn/).

### Citation

If you find this project useful in your research, please consider citing:

```latex
@misc{MindSpore Pose 2022,
    title={{MindSpore Pose}:MindSpore Pose Toolbox and Benchmark},
    author={MindSpore Vision Contributors},
    howpublished = {\url{https://github.com/mindspore-lab/mindpose/}},
    year={2022}
}
```

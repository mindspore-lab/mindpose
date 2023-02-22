import mindspore as ms
import numpy as np

from mindpose.models.backbones.resnet import resnet101, resnet152, resnet50
from mindspore import Tensor


def test_resnet50_forward():
    ms.set_context(mode=ms.GRAPH_MODE)

    net = resnet50(in_channels=3)
    inputs = Tensor(np.random.rand(4, 3, 32, 32), dtype=ms.float32)
    output = net(inputs)
    assert output.shape == (4, 2048, 1, 1)


def test_resnet101_forward():
    ms.set_context(mode=ms.GRAPH_MODE)

    net = resnet101(in_channels=3)
    inputs = Tensor(np.random.rand(4, 3, 32, 32), dtype=ms.float32)
    output = net(inputs)
    assert output.shape == (4, 2048, 1, 1)


def test_resnet152_forward():
    ms.set_context(mode=ms.GRAPH_MODE)

    net = resnet152(in_channels=3)
    inputs = Tensor(np.random.rand(4, 3, 32, 32), dtype=ms.float32)
    output = net(inputs)
    assert output.shape == (4, 2048, 1, 1)

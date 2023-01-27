import mindspore as ms
import numpy as np

from mindpose.models.backbones.hrnet import hrnet_w32, hrnet_w48
from mindspore import Tensor

ms.set_context(mode=ms.GRAPH_MODE)


def test_hrnet_w32_forward():
    net = hrnet_w32(in_channels=3)
    inputs = Tensor(np.random.rand(4, 3, 32, 32), dtype=ms.float32)
    output = net(inputs)
    assert output.shape == (4, 32, 8, 8)


def test_hrnet_w48_forward():
    net = hrnet_w48(in_channels=3)
    inputs = Tensor(np.random.rand(4, 3, 32, 32), dtype=ms.float32)
    output = net(inputs)
    assert output.shape == (4, 48, 8, 8)

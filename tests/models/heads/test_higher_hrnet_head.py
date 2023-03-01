import mindspore as ms
import numpy as np

from mindpose.models.heads.higher_hrnet_head import HigherHRNetHead
from mindspore import Tensor


def test_higher_hrnet_head_forward():
    ms.set_context(mode=ms.GRAPH_MODE)

    net = HigherHRNetHead(in_channels=32, num_joints=17)
    inputs = Tensor(np.random.random((4, 32, 8, 8)), dtype=ms.float32)
    output = net(inputs)
    assert output[0].shape == (4, 34, 8, 8)
    assert output[1].shape == (4, 17, 16, 16)


def test_higher_hrnet_head_forward_without_concat():
    ms.set_context(mode=ms.GRAPH_MODE)

    net = HigherHRNetHead(in_channels=32, num_joints=17, cat_outputs=[False])
    inputs = Tensor(np.random.random((4, 32, 8, 8)), dtype=ms.float32)
    output = net(inputs)
    assert output[0].shape == (4, 34, 8, 8)
    assert output[1].shape == (4, 17, 16, 16)


def test_higher_hrnet_head_forward_without_tag_per_joint():
    ms.set_context(mode=ms.GRAPH_MODE)

    net = HigherHRNetHead(in_channels=32, num_joints=17, tag_per_joint=False)
    inputs = Tensor(np.random.random((4, 32, 8, 8)), dtype=ms.float32)
    output = net(inputs)
    assert output[0].shape == (4, 18, 8, 8)
    assert output[1].shape == (4, 17, 16, 16)

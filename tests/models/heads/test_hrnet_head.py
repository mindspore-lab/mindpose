import mindspore as ms
import numpy as np

from mindpose.models.heads.hrnet_head import HRNetHead
from mindspore import Tensor


def test_hrnet_head_forward():
    ms.set_context(mode=ms.GRAPH_MODE)

    net = HRNetHead(in_channels=32, num_joints=17)
    inputs = Tensor(np.random.random((4, 32, 8, 8)), dtype=ms.float32)
    output = net(inputs)
    assert output.shape == (4, 17, 8, 8)

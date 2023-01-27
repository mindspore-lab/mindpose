import mindspore as ms
import numpy as np

from mindpose.models.heads.simple_baseline_head import SimpleBaselineHead
from mindspore import Tensor

ms.set_context(mode=ms.GRAPH_MODE)


def test_simple_baseline_head_forward():
    net = SimpleBaselineHead(in_channels=32, num_joints=17)
    inputs = Tensor(np.random.random((4, 32, 8, 8)), dtype=ms.float32)
    output = net(inputs)
    assert output.shape == (4, 17, 64, 64)

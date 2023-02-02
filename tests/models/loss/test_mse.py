import mindspore as ms
import numpy as np

from mindpose.models.loss.mse import JointsMSELoss
from mindspore import Tensor

ms.set_context(mode=ms.GRAPH_MODE)


def test_joint_mse():
    criterion = JointsMSELoss()
    pred = Tensor(np.random.random((4, 12, 32, 32)), dtype=ms.float32)
    target = Tensor(np.random.random((4, 12, 32, 32)), dtype=ms.float32)
    loss = criterion(pred, target)
    assert loss.size == 1


def test_joint_mse_with_target_weight():
    criterion = JointsMSELoss(use_target_weight=True)
    pred = Tensor(np.random.random((4, 12, 32, 32)), dtype=ms.float32)
    target = Tensor(np.random.random((4, 12, 32, 32)), dtype=ms.float32)
    target_weight = Tensor(np.random.random((4, 12)), dtype=ms.float32)
    loss = criterion(pred, target, target_weight)
    assert loss.size == 1

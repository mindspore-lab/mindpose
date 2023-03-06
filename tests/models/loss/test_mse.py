import mindspore as ms
import numpy as np

from mindpose.models.loss.mse import JointsMSELoss, JointsMSELossWithMask
from mindspore import Tensor


def test_joint_mse():
    ms.set_context(mode=ms.GRAPH_MODE)

    criterion = JointsMSELoss()
    pred = Tensor(np.random.random((4, 12, 32, 32)), dtype=ms.float32)
    target = Tensor(np.random.random((4, 12, 32, 32)), dtype=ms.float32)
    loss = criterion(pred, target)
    assert loss.size == 1


def test_joint_mse_with_target_weight():
    ms.set_context(mode=ms.GRAPH_MODE)

    criterion = JointsMSELoss(use_target_weight=True)
    pred = Tensor(np.random.random((4, 12, 32, 32)), dtype=ms.float32)
    target = Tensor(np.random.random((4, 12, 32, 32)), dtype=ms.float32)
    target_weight = Tensor(np.random.random((4, 12)), dtype=ms.float32)
    loss = criterion(pred, target, target_weight)
    assert loss.size == 1


def test_joint_mse_with_mask():
    ms.set_context(mode=ms.GRAPH_MODE)

    criterion = JointsMSELossWithMask()
    pred = Tensor(np.random.random((4, 12, 32, 32)), dtype=ms.float32)
    target = Tensor(np.random.random((4, 12, 32, 32)), dtype=ms.float32)
    mask = Tensor(np.random.randint(2, size=(4, 32, 32)), dtype=ms.uint8)
    loss = criterion(pred, target, mask)
    assert loss.size == 1

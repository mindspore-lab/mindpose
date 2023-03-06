import mindspore as ms
import numpy as np

from mindpose.models.loss.multi_loss import AEMultiLoss
from mindspore import Tensor


def test_ae_multi_loss():
    ms.set_context(mode=ms.GRAPH_MODE)

    num_joints = 3
    criterion = AEMultiLoss(
        num_joints=num_joints,
        with_ae_loss=[True, False],
        stage_sizes=[(16, 16), (32, 32)],
    )
    pred = [
        Tensor(np.random.random((4, 2 * num_joints, 16, 16)), dtype=ms.float32),
        Tensor(np.random.random((4, num_joints, 32, 32)), dtype=ms.float32),
    ]
    target = Tensor(np.random.random((4, 2, num_joints, 32, 32)), dtype=ms.float32)
    mask = Tensor(np.random.randint(2, size=(4, 2, 32, 32)), dtype=ms.uint8)
    tag_mask = Tensor(
        np.random.randint(2, size=(4, 1, 5, num_joints, 16, 16)), dtype=ms.uint8
    )
    loss = criterion(pred, target, mask, tag_mask)
    assert loss.size == 1

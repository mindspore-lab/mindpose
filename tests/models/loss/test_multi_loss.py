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
    preds = [
        Tensor(np.random.random((4, 2 * num_joints, 16, 16)), dtype=ms.float32),
        Tensor(np.random.random((4, num_joints, 32, 32)), dtype=ms.float32),
    ]
    target = Tensor(np.random.random((4, 2, num_joints, 32, 32)), dtype=ms.float32)
    mask = Tensor(np.random.randint(2, size=(4, 2, 32, 32)), dtype=ms.uint8)
    tag_ind_1 = np.random.randint(16 * 16, size=(4, 2, 5, num_joints, 1))
    tag_ind_2 = np.random.randint(2, size=(4, 2, 5, num_joints, 1))
    tag_ind = np.concatenate([tag_ind_1, tag_ind_2], axis=-1)
    tag_ind = Tensor(tag_ind, dtype=ms.int32)

    loss = criterion(preds, target, mask, tag_ind)
    assert loss.size == 1


def test_ae_multi_loss_with_tag_per_joint_is_false():
    ms.set_context(mode=ms.GRAPH_MODE)

    num_joints = 3
    criterion = AEMultiLoss(
        num_joints=num_joints,
        with_ae_loss=[True, False],
        stage_sizes=[(16, 16), (32, 32)],
        tag_per_joint=False,
    )
    preds = [
        Tensor(np.random.random((4, num_joints + 1, 16, 16)), dtype=ms.float32),
        Tensor(np.random.random((4, num_joints, 32, 32)), dtype=ms.float32),
    ]
    target = Tensor(np.random.random((4, 2, num_joints, 32, 32)), dtype=ms.float32)
    mask = Tensor(np.random.randint(2, size=(4, 2, 32, 32)), dtype=ms.uint8)
    tag_ind_1 = np.random.randint(16 * 16, size=(4, 2, 5, 1))
    tag_ind_2 = np.random.randint(2, size=(4, 2, 5, 1))
    tag_ind = np.concatenate([tag_ind_1, tag_ind_2], axis=-1)
    tag_ind = Tensor(tag_ind, dtype=ms.int32)

    loss = criterion(preds, target, mask, tag_ind)
    assert loss.size == 1

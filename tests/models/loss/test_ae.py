import mindspore as ms
import numpy as np

from mindpose.models.loss.ae import AELoss
from mindspore import Tensor


def test_ae():
    ms.set_context(mode=ms.GRAPH_MODE)

    criterion = AELoss(reduce=True)

    # pred: [N, K, H, W]
    pred = Tensor(
        np.arange(0, 2 * 3 * 4 * 4).reshape(2, 3, 4, 4) * 0.01, dtype=ms.float32
    )

    # target: [N, M, K, 2]
    target_1 = np.random.randint(16, size=(2, 5, 3, 1), dtype=np.int32)
    target_2 = np.random.randint(2, size=(2, 5, 3, 1), dtype=np.int32)
    target = np.concatenate((target_1, target_2), axis=-1)
    target = Tensor(target)

    loss = criterion(pred, target)
    assert loss.size == 1


def test_ae_with_tag_per_joint_is_false():
    ms.set_context(mode=ms.GRAPH_MODE)

    criterion = AELoss(reduce=True, tag_per_joint=False)

    # pred: [N, H, W]
    pred = Tensor(np.arange(0, 2 * 1 * 4 * 4).reshape(2, 4, 4) * 0.01, dtype=ms.float32)

    # target: [N, M, 2]
    target_1 = np.random.randint(16, size=(2, 5, 1), dtype=np.int32)
    target_2 = np.random.randint(2, size=(2, 5, 1), dtype=np.int32)
    target = np.concatenate((target_1, target_2), axis=-1)
    target = Tensor(target)

    loss = criterion(pred, target)
    assert loss.size == 1

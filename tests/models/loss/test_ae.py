import mindspore as ms
import numpy as np

from mindpose.models.loss.ae import AELoss
from mindspore import Tensor


def test_ae_with_reduce_is_true():
    ms.set_context(mode=ms.GRAPH_MODE)

    criterion = AELoss(reduce=True)

    # pred: [N, K, H, W]
    pred = Tensor(
        np.arange(0, 2 * 3 * 4 * 4).reshape(2, 3, 4, 4) * 0.01, dtype=ms.float32
    )

    # target: [N, M, K, H, W]
    target = np.random.randint(2, size=(2, 5, 3, 4, 4), dtype=np.uint8)
    target = Tensor(target)

    loss = criterion(pred, target)
    assert loss.size == 1

import mindspore as ms
import numpy as np

from mindpose.models.decoders.bottom_up_decoder import BottomUpHeatMapAEDecoder
from mindspore import Tensor


def test_heatmap_ae_decoder():
    ms.set_context(mode=ms.GRAPH_MODE)

    decoder = BottomUpHeatMapAEDecoder(max_num=30)
    heatmap_1 = Tensor(np.random.random((8, 34, 32, 32)), dtype=ms.float32)
    heatmap_2 = Tensor(np.random.random((8, 17, 64, 64)), dtype=ms.float32)
    mask = Tensor(np.random.randint(2, size=(8, 128, 128)), dtype=ms.uint8)

    val_k, tag_k, ind_k = decoder([heatmap_1, heatmap_2], mask)
    assert val_k.shape == (8, 17, 30)
    assert tag_k.shape == (8, 17, 30, 1)
    assert ind_k.shape == (8, 17, 30, 2)

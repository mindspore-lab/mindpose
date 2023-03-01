import mindspore as ms
import numpy as np

from mindpose.models.decoders.top_down_decoder import TopDownHeatMapDecoder
from mindspore import Tensor


def test_top_down_decoder():
    ms.set_context(mode=ms.GRAPH_MODE)

    decoder = TopDownHeatMapDecoder()
    heatmap = Tensor(np.random.random((8, 17, 48, 64)), dtype=ms.float32)
    center = Tensor(np.random.uniform(low=0, high=400, size=(8, 2)), dtype=ms.float32)
    scale = Tensor(np.random.uniform(low=0, high=3, size=(8, 2)), dtype=ms.float32)
    score = Tensor(np.random.random((8)), dtype=ms.float32)

    all_preds, all_boxes = decoder(heatmap, center, scale, score)
    assert all_preds.shape == (8, 17, 3)
    assert all_boxes.shape == (8, 6)


def test_top_down_decoder_with_shift_coordinate():
    ms.set_context(mode=ms.GRAPH_MODE)

    decoder = TopDownHeatMapDecoder(shift_coordinate=True)
    heatmap = Tensor(np.random.random((8, 17, 48, 64)), dtype=ms.float32)
    center = Tensor(np.random.uniform(low=0, high=400, size=(8, 2)), dtype=ms.float32)
    scale = Tensor(np.random.uniform(low=0, high=3, size=(8, 2)), dtype=ms.float32)
    score = Tensor(np.random.random((8)), dtype=ms.float32)

    all_preds, all_boxes = decoder(heatmap, center, scale, score)
    assert all_preds.shape == (8, 17, 3)
    assert all_boxes.shape == (8, 6)


def test_top_down_decoder_with_udp_refine():
    ms.set_context(mode=ms.GRAPH_MODE)

    decoder = TopDownHeatMapDecoder(use_udp=True, dark_udp_refine=True)
    heatmap = Tensor(np.random.random((8, 17, 48, 64)), dtype=ms.float32)
    center = Tensor(np.random.uniform(low=0, high=400, size=(8, 2)), dtype=ms.float32)
    scale = Tensor(np.random.uniform(low=0, high=3, size=(8, 2)), dtype=ms.float32)
    score = Tensor(np.random.random((8)), dtype=ms.float32)

    all_preds, all_boxes = decoder(heatmap, center, scale, score)
    assert all_preds.shape == (8, 17, 3)
    assert all_boxes.shape == (8, 6)

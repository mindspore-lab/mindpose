from typing import Any, Dict, List, Optional, Tuple

import mindspore as ms
from mindspore import Tensor
from mindspore.dataset import Dataset
from tqdm import tqdm

from ...models import EvalNet

from ...models.decoders import TopDownHeatMapDecoder
from .inferencer import Inferencer


class TopDownHeatMapInferencer(Inferencer):
    def __init__(
        self,
        net: EvalNet,
        config: Optional[Dict[str, Any]] = None,
        progress_bar: bool = False,
        decoder: Optional[TopDownHeatMapDecoder] = None,
    ) -> None:
        super().__init__(net, config=config)
        self.progress_bar = progress_bar
        self.decoder = decoder

        if (
            self._inference_cfg["hflip_tta"]
            and not self._inference_cfg["has_heatmap_output"]
        ):
            raise ValueError("flip TTA need heatmap output.")

        if self.decoder is None and self._inference_cfg["hflip_tta"]:
            raise ValueError("Decoder must be provided for flip TTA")

    def load_inference_cfg(self) -> Dict[str, Any]:
        """Loading the annoation info from the config file"""
        inference_cfg = dict()

        inference_cfg["has_heatmap_output"] = self.config.get(
            "has_heatmap_output", True
        )
        inference_cfg["hflip_tta"] = self.config.get("hflip_tta", False)
        inference_cfg["shift_heatmap"] = self.config.get("shift_heatmap", False)

        # TODO: read array from config
        inference_cfg["flip_pairs"] = [
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
            [9, 10],
            [11, 12],
            [13, 14],
            [15, 16],
        ]
        return inference_cfg

    def __call__(self, dataset: Dataset) -> List[Dict[str, Any]]:
        outputs = list()
        for data in tqdm(
            dataset.create_dict_iterator(num_epochs=1),
            total=dataset.get_dataset_size(),
            disable=not self.progress_bar,
        ):
            if self._inference_cfg["has_heatmap_output"]:
                (preds, boxes), heatmap = self.net(
                    data["image"], data["center"], data["scale"], data["bbox_scores"]
                )
            else:
                preds, boxes = self.net(
                    data["image"], data["center"], data["scale"], data["bbox_scores"]
                )

            if self._inference_cfg["hflip_tta"]:
                flipped_image = ms.numpy.flip(data["image"], axis=3)
                _, flipped_heatmap = self.net(
                    flipped_image, data["center"], data["scale"], data["bbox_scores"]
                )
                flipped_heatmap = flip_back(
                    flipped_heatmap, self._inference_cfg["flip_pairs"]
                )

                if self._inference_cfg["shift_heatmap"]:
                    flipped_heatmap[:, :, :, 1:] = flipped_heatmap[:, :, :, :-1]

                heatmap = (heatmap + flipped_heatmap) * 0.5
                preds, boxes = self.decoder(
                    heatmap, data["center"], data["scale"], data["bbox_scores"]
                )

            preds = preds.asnumpy()
            boxes = boxes.asnumpy()
            image_paths = data["image_file"].asnumpy()
            bbox_ids = data["bbox_ids"].asnumpy()

            for pred, box, image_path, bbox_id in zip(
                preds, boxes, image_paths, bbox_ids
            ):
                record = dict(
                    pred=pred.tolist(),
                    box=box.tolist(),
                    image_path=image_path.tolist(),
                    bbox_id=bbox_id.tolist(),
                )
                outputs.append(record)
        return outputs


def flip_back(flipped_heatmap: Tensor, flip_pairs: List[Tuple[int, int]]) -> Tensor:
    """Flip the flipped heatmaps back to the original form."""
    shape_ori = flipped_heatmap.shape
    channels = 1
    flipped_heatmap = flipped_heatmap.reshape(
        shape_ori[0], -1, channels, shape_ori[2], shape_ori[3]
    )
    flipped_heatmap_back = flipped_heatmap.copy()

    # Swap left-right parts
    for left, right in flip_pairs:
        flipped_heatmap_back[:, left, ...] = flipped_heatmap[:, right, ...]
        flipped_heatmap_back[:, right, ...] = flipped_heatmap[:, left, ...]
    flipped_heatmap_back = flipped_heatmap_back.reshape(shape_ori)
    # Flip horizontally
    flipped_heatmap_back = flipped_heatmap_back[..., ::-1]
    return flipped_heatmap_back

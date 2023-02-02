from typing import Any, Dict, List, Optional, Tuple

import mindspore as ms
import numpy as np
from mindspore import Tensor
from mindspore.dataset import Dataset
from tqdm import tqdm

from ...models import EvalNet
from ...models.decoders import TopDownHeatMapDecoder
from ...register import register
from .inferencer import Inferencer


@register("inferencer", extra_name="topdown_heatmap")
class TopDownHeatMapInferencer(Inferencer):
    """Create an inference engine for Topdown heatmap based method.
    It runs the inference on the entire dataset and outputs a list of records.

    Args:
        net: Network for evaluation
        config: Method-specific configuration. Default: None
        progress_bar: Display the progress bar during inferencing. Default: False
        decoder: Decoder cell. It is used for hflip TTA. Default: None

    Inputs:
        | dataset: Dataset

    Outputs:
        | records: List of inference records.
    """

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

        if self.decoder is None and self._inference_cfg["hflip_tta"]:
            raise ValueError("Decoder must be provided for flip TTA")

    def load_inference_cfg(self) -> Dict[str, Any]:
        """Loading the inference config, where the returned config must be a dictionary
        which stores the configuration of the engine, such as the using TTA, etc.

        Returns:
            Inference configurations
        """
        inference_cfg = dict()

        inference_cfg["has_heatmap_output"] = self.config["has_heatmap_output"]
        inference_cfg["hflip_tta"] = self.config["hflip_tta"]
        inference_cfg["shift_heatmap"] = self.config["shift_heatmap"]
        inference_cfg["flip_pairs"] = np.array(self.config["flip_pairs"])

        if inference_cfg["hflip_tta"] and not inference_cfg["has_heatmap_output"]:
            raise ValueError("flip TTA need heatmap output.")

        return inference_cfg

    def infer(self, dataset: Dataset) -> List[Dict[str, Any]]:
        """Running the inference on the dataset. And return a list of records.
        Normally, in order to be compatible with the evaluator engine,
        each record should contains the following keys:

        Keys:
            | pred: The predicted coordindate, in shape [M, 3(x_coord, y_coord, score)]
            | box: The coor bounding boxes, each record contains
                (center_x, center_y, scale_x, scale_y, area, bounding box score)
            | image_path: The path of the image
            | bbox_id: Bounding box ID

        Args:
            dataset: Dataset for inferencing

        Returns:
            List of inference results
        """
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
                flipped_heatmap = _flip_back(
                    flipped_heatmap, self._inference_cfg["flip_pairs"].tolist()
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


def _flip_back(flipped_heatmap: Tensor, flip_pairs: List[Tuple[int, int]]) -> Tensor:
    """Flip the flipped heatmaps back to the original form."""
    flipped_heatmap_back = flipped_heatmap.copy()

    # Swap left-right parts
    for left, right in flip_pairs:
        flipped_heatmap_back[:, left, ...] = flipped_heatmap[:, right, ...]
        flipped_heatmap_back[:, right, ...] = flipped_heatmap[:, left, ...]

    # Flip horizontally
    flipped_heatmap_back = flipped_heatmap_back[..., ::-1]
    return flipped_heatmap_back

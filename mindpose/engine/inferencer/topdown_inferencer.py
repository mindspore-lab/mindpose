from typing import Any, Dict, List, Optional, Tuple, Union

import mindspore as ms
import mindspore.nn as nn
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

        if (
            self._inference_cfg["hflip_tta"]
            and not self._inference_cfg["has_heatmap_output"]
        ):
            raise ValueError("flip TTA need heatmap output.")

        if self._inference_cfg["hflip_tta"]:
            self._multi_run_net = _MultiRunNet(
                self.net,
                self.decoder,
                self._inference_cfg["flip_index"],
                shift_heatmap=self._inference_cfg["shift_heatmap"],
            )
            self._multi_run_net.set_train(False)
        else:
            self._multi_run_net = None

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

        flip_index = np.array(self.config["flip_pairs"])[:, ::-1].flatten()
        flip_index = np.insert(flip_index, 0, 0)
        inference_cfg["flip_index"] = flip_index

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
            if self._inference_cfg["hflip_tta"]:
                preds, boxes = self._multi_run_net(
                    data["image"], data["center"], data["scale"], data["bbox_scores"]
                )
            else:
                if self._inference_cfg["has_heatmap_output"]:
                    (preds, boxes), _ = self.net(
                        data["image"],
                        data["center"],
                        data["scale"],
                        data["bbox_scores"],
                    )
                else:
                    preds, boxes = self.net(
                        data["image"],
                        data["center"],
                        data["scale"],
                        data["bbox_scores"],
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


class _MultiRunNet(nn.Cell):
    """Running the inference for multiple times with horizontal TTA."""

    def __init__(
        self,
        net: Inferencer,
        decoder: TopDownHeatMapDecoder,
        flip_index: Union[np.ndarray, Tensor],
        shift_heatmap: bool = False,
    ) -> None:
        super().__init__()
        self.net = net
        self.decoder = decoder
        self.shift_heatmap = shift_heatmap
        if isinstance(flip_index, np.ndarray):
            self.flip_index = Tensor(flip_index)
        else:
            self.flip_index = flip_index

    def construct(
        self, image: Tensor, center: Tensor, scale: Tensor, score: Tensor
    ) -> Tuple[Tensor, Tensor]:
        _, heatmap = self.net(image, center, scale, score)
        flipped_image = ms.numpy.flip(image, axis=3)
        _, flipped_heatmap = self.net(flipped_image, center, scale, score)
        flipped_heatmap = self._flip_back(flipped_heatmap)

        if self.shift_heatmap:
            flipped_heatmap = self._shift_heatmap(flipped_heatmap)

        final_heatmap = (heatmap + flipped_heatmap) * 0.5
        preds = self.decoder(final_heatmap, center, scale, score)
        return preds

    def _flip_back(self, flipped_heatmap: Tensor) -> Tensor:
        flipped_heatmap_back = flipped_heatmap[:, self.flip_index, ...]
        flipped_heatmap_back = flipped_heatmap_back[..., ::-1]
        return flipped_heatmap_back

    def _shift_heatmap(self, heatmap: Tensor) -> Tensor:
        heatmap[..., 1:] = heatmap[..., :-1]
        return heatmap

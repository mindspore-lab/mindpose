from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from mindspore import Tensor
from mindspore.dataset import Dataset
from tqdm import tqdm

from ...data.transform.utils import transform_keypoints
from ...models import EvalNet
from ...models.decoders import BottomUpHeatMapAEDecoder
from ...register import register
from ...utils.match import match_by_tag
from .inferencer import Inferencer


@register("inferencer", extra_name="bottomup_heatmap_ae")
class BottomUpHeatMapAEInferencer(Inferencer):
    """Create an inference engine for bottom-up heatmap with associative embedding
    based method. It runs the inference on the entire dataset and outputs a list of
    records.

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
        decoder: Optional[BottomUpHeatMapAEDecoder] = None,
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
        inference_cfg["joint_order"] = self.config["joint_order"]
        inference_cfg["vis_thr"] = float(self.config["vis_thr"])
        inference_cfg["ignore_too_much"] = self.config["ignore_too_much"]
        inference_cfg["use_detection_val"] = self.config["use_detection_val"]
        inference_cfg["tag_thr"] = float(self.config["tag_thr"])
        inference_cfg["pixel_std"] = float(self.config["pixel_std"])
        inference_cfg["downsample_scale"] = self.config["downsample_scale"]

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
                preds, heatmap = self.net(data["image"], data["mask"])
            else:
                preds = self.net(data["image"], data["mask"])

            preds, scores = self._parse(*preds)

            center = data["center"].asnumpy()
            scale = data["scale"].asnumpy()
            image_shape = data["image_shape"].asnumpy()
            image_paths = data["image_file"].asnumpy()

            preds = transform_keypoints(
                preds,
                center,
                scale,
                image_shape / self._inference_cfg["downsample_scale"],
                pixel_std=self._inference_cfg["pixel_std"],
            )
            for pred, score, image_path in zip(preds, scores, image_paths):
                record = dict(pred=pred, score=score, image_path=image_path)
                outputs.append(record)
        return outputs

    def _parse(
        self, val_k: Tensor, tag_k: Tensor, ind_k: Tensor
    ) -> Tuple[List[np.ndarray], List[List[float]]]:
        grouped = self._match(val_k, tag_k, ind_k)
        # calculate the score for each instance
        scores = list()
        for x in grouped:
            scores.append([y[:, 2].mean() for y in x])
        return grouped, scores

    def _match(self, val_k: Tensor, tag_k: Tensor, ind_k: Tensor) -> List[np.ndarray]:
        func = partial(
            match_by_tag,
            joint_order=self._inference_cfg["joint_order"],
            vis_thr=self._inference_cfg["vis_thr"],
            tag_thr=self._inference_cfg["tag_thr"],
            ignore_too_much=self._inference_cfg["ignore_too_much"],
            use_detection_val=self._inference_cfg["use_detection_val"],
        )
        return list(
            map(
                func,
                val_k.asnumpy(),
                tag_k.asnumpy(),
                ind_k.asnumpy(),
            )
        )

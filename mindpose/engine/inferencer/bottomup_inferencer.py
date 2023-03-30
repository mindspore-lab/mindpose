from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import mindspore as ms
import mindspore.nn as nn
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

        if (
            self._inference_cfg["hflip_tta"]
            and not self._inference_cfg["has_heatmap_output"]
        ):
            raise ValueError("flip TTA need heatmap output.")

        if self._inference_cfg["hflip_tta"]:
            self._multi_run_net = _MultiRunNet(
                self.net, self.decoder, self._inference_cfg["flip_index"]
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
        inference_cfg["joint_order"] = self.config["joint_order"]
        inference_cfg["vis_thr"] = float(self.config["vis_thr"])
        inference_cfg["ignore_too_much"] = self.config["ignore_too_much"]
        inference_cfg["use_rounded_norm"] = self.config["use_rounded_norm"]
        inference_cfg["tag_thr"] = float(self.config["tag_thr"])
        inference_cfg["pixel_std"] = float(self.config["pixel_std"])
        inference_cfg["downsample_scale"] = self.config["downsample_scale"]
        inference_cfg["refine_missing_joint"] = self.config["refine_missing_joint"]

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
                preds = self._multi_run_net(data["image"], data["mask"])
            else:
                if self._inference_cfg["has_heatmap_output"]:
                    preds, _ = self.net(data["image"], data["mask"])
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
        self,
        val_k: Tensor,
        tag_k: Tensor,
        ind_k: Tensor,
        heatmap: Tensor,
        tagging_heatmap: Tensor,
    ) -> Tuple[List[np.ndarray], List[List[float]]]:
        """Output the final result by post-processings."""
        keypoints = self._match(val_k, tag_k, ind_k)

        # calculate the score for each instance
        scores = list()
        for x in keypoints:
            scores.append([y[:, 2].mean() for y in x])

        # add missing joints
        if self._inference_cfg["refine_missing_joint"]:
            heatmap = heatmap.asnumpy()
            tagging_heatmap = tagging_heatmap.asnumpy()
            for i in range(len(keypoints)):
                for j in range(len(keypoints[i])):
                    keypoints[i][j] = self._refine_missing(
                        heatmap[i], tagging_heatmap[i], keypoints[i][j]
                    )

        return keypoints, scores

    def _match(self, val_k: Tensor, tag_k: Tensor, ind_k: Tensor) -> List[np.ndarray]:
        """Match the result by tag."""
        func = partial(
            match_by_tag,
            joint_order=self._inference_cfg["joint_order"],
            vis_thr=self._inference_cfg["vis_thr"],
            tag_thr=self._inference_cfg["tag_thr"],
            ignore_too_much=self._inference_cfg["ignore_too_much"],
            use_rounded_norm=self._inference_cfg["use_rounded_norm"],
        )
        return list(
            map(
                func,
                val_k.asnumpy(),
                tag_k.asnumpy(),
                ind_k.asnumpy(),
            )
        )

    def _refine_missing(
        self,
        heatmap: np.ndarray,
        tagging_heatmap: np.ndarray,
        keypoints: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        heatmap: K x H x W
        tagging_heatmap: K x H x W x L
        keypoints: K x 4
        """
        K, H, W = heatmap.shape

        # calculate the mean tag for single instance
        tags = []
        location = keypoints[:, :2].astype(np.int32)
        for i in range(K):
            if keypoints[i, 2] > 0:
                x, y = location[i]
                tags.append(tagging_heatmap[i, y, x])
        mean_tag = np.mean(tags, axis=0)

        # calculate the L2-distance of the tagging heatmap and mean tag
        dist = tagging_heatmap - mean_tag[None, None, None, :]
        dist = np.linalg.norm(dist, axis=3)
        dist = np.round(dist)

        # find the location maximize the heatmap - L2 distance of the tag
        dist_2 = (heatmap - dist).reshape(K, -1)
        max_loc = np.argmax(dist_2, axis=1)
        ys, xs = np.unravel_index(max_loc, (H, W))
        ys_int, xs_int = ys.copy(), xs.copy()
        xs = xs.astype(np.float32)
        ys = ys.astype(np.float32)
        # offset by 0.5
        xs += 0.5
        ys += 0.5

        # shifted by +- 0.25
        for i in range(heatmap.shape[0]):
            xx = xs_int[i]
            yy = ys_int[i]

            if heatmap[i, yy, min(xx + 1, W - 1)] > heatmap[i, yy, max(xx - 1, 0)]:
                xs[i] += 0.25
            else:
                xs[i] -= 0.25

            if heatmap[i, min(yy + 1, H - 1), xx] > heatmap[i, max(0, yy - 1), xx]:
                ys[i] += 0.25
            else:
                ys[i] -= 0.25

        # add keypoint if it is not detected
        vals = heatmap[np.arange(K), ys_int, xs_int]
        keypoints_full = np.stack((xs, ys, vals), axis=1)
        for i in range(K):
            if keypoints_full[i, 2] > 0 and keypoints[i, 2] == 0:
                keypoints[i, :3] = keypoints_full[i]

        return keypoints


class _MultiRunNet(nn.Cell):
    """Running the inference for multiple times with horizontal TTA."""

    def __init__(
        self,
        net: EvalNet,
        decoder: BottomUpHeatMapAEDecoder,
        flip_index: Union[np.ndarray, Tensor],
    ) -> None:
        super().__init__()
        self.net = net
        self.decoder = decoder
        if isinstance(flip_index, np.ndarray):
            self.flip_index = Tensor(flip_index)
        else:
            self.flip_index = flip_index

    def construct(self, image: Tensor, mask: Tensor) -> List[Tensor]:
        _, raw_output = self.net(image, mask)
        flipped_image = ms.numpy.flip(image, axis=3)
        _, flipped_raw_output = self.net(flipped_image, mask)

        heatmap, tagging_heatmap = self.decoder.decouple_output(raw_output)
        flipped_heatmap, flipped_tagging_heatmap = self.decoder.decouple_output(
            flipped_raw_output
        )
        flipped_heatmap = self._flip_back(flipped_heatmap)
        flipped_tagging_heatmap = self._flip_back(flipped_tagging_heatmap)

        final_heatmap = (heatmap + flipped_heatmap) * 0.5
        final_tagging_heatmap = list()
        for x in tagging_heatmap:
            final_tagging_heatmap.append(x)
        for x in flipped_tagging_heatmap:
            final_tagging_heatmap.append(x)

        preds = self.decoder.decode(final_heatmap, final_tagging_heatmap, mask)
        return preds

    def _flip_back(self, flipped_heatmaps: List[Tensor]) -> List[Tensor]:
        output = list()
        for flipped_heatmap in flipped_heatmaps:
            flipped_heatmap_back = flipped_heatmap[:, self.flip_index, ...]
            flipped_heatmap_back = flipped_heatmap_back[..., ::-1]
            output.append(flipped_heatmap_back)
        return output

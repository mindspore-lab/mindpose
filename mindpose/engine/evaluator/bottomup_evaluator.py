import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

import numpy as np

from mindpose.utils.nms import oks_nms, soft_oks_nms
from ...register import register
from .evaluator import Evaluator


@register("evaluator", extra_name="bottomup")
class BottomUpEvaluator(Evaluator):
    """Create an evaluator based on BottomUp method. It performs the model
    evaluation based on the inference result (a list of records), and
    outputs with the metirc result.

    Args:
        annotation_file: Path of the annotation file. It only supports COCO-format.
        metric: Supported metrics. Default: "AP"
        num_joints: Number of joints. Default: 17
        config: Method-specific configuration. Default: None
        remove_result_file: Remove the cached result file
            after evaluation. Default: True
        result_path: Path of the result file. Default: "./result_keypoints.json"

    Inputs:
        | inference_result: Inference result from inference engine

    Outputs:
        | evaluation_result: Evaluation result based on the metric
    """

    SUPPORT_METRICS = {"AP"}

    def __init__(
        self,
        annotation_file: str,
        metric: Union[str, List[str]] = "AP",
        num_joints: int = 17,
        config: Optional[Dict[str, Any]] = None,
        remove_result_file: bool = True,
        result_path: str = "./result_keypoints.json",
    ) -> None:
        super().__init__(
            annotation_file, metric=metric, num_joints=num_joints, config=config
        )
        self.remove_result_file = remove_result_file
        self.result_path = result_path

    def load_evaluation_cfg(self) -> Dict[str, Any]:
        """Loading the evaluation config, where the returned config must be a dictionary
        which stores the configuration of the engine, such as the using soft-nms, etc.

        Returns:
            Evaluation configurations
        """
        evaluation_cfg = dict()
        evaluation_cfg["oks_thr"] = self.config["oks_thr"]
        evaluation_cfg["use_nms"] = self.config["use_nms"]
        evaluation_cfg["soft_nms"] = self.config["soft_nms"]
        evaluation_cfg["sigmas"] = np.array(self.config["sigmas"])
        return evaluation_cfg

    def eval(self, inference_result: Dict[str, Any]) -> Dict[str, Any]:
        """Running the evaluation base on the inference result. Output the metric result.

        Args:
            inference_result: List of inference records

        Returns:
            metric result. Such as AP.5, etc.
        """
        kpts = defaultdict(list)

        for record in inference_result:
            image_path = record["image_path"]
            image_id = self.name2id[os.path.basename(image_path)]
            for kpt, score in zip(record["pred"], record["score"]):
                # kpt: K x 4
                area = (np.max(kpt[:, 0]) - np.min(kpt[:, 0])) * (
                    np.max(kpt[:, 1]) - np.min(kpt[:, 1])
                )
                kpts[image_id].append(
                    {
                        "keypoints": kpt[:, :3],
                        "score": score,
                        "image_id": image_id,
                        "area": area,
                    }
                )

        # oks nms if necessary
        oks_thr = self._evaluation_cfg["oks_thr"]
        valid_kpts = []
        for image_id in kpts.keys():
            img_kpts = kpts[image_id]
            if self._evaluation_cfg["use_nms"]:
                nms = soft_oks_nms if self._evaluation_cfg["soft_nms"] else oks_nms
                keep = nms(
                    img_kpts, oks_thr, sigmas=np.asarray(self._evaluation_cfg["sigmas"])
                )
                valid_kpts.append([img_kpts[_keep] for _keep in keep])
            else:
                valid_kpts.append(img_kpts)

        self._write_coco_keypoint_results(valid_kpts, self.result_path)

        info_str = self._do_python_keypoint_eval(self.result_path)
        name_value = dict(info_str)

        for name in self.metrics:
            if name not in name_value:
                raise ValueError(
                    f"`{name}` is not in the returned result `{name_value.keys()}`"
                )

        # clean the temporary result
        if self.remove_result_file:
            os.remove(self.result_path)

        return name_value

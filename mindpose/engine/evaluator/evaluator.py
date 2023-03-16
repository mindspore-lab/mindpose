import json
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class Evaluator:
    """Create an evaluator engine. It performs the model evaluation based on the
    inference result (a list of records), and outputs with the metirc result.

    Args:
        annotation_file: Path of the annotation file. It only supports COCO-format now.
        metric: Supported metrics. Default: "AP"
        num_joints: Number of joints. Default: 17
        config: Method-specific configuration. Default: None

    Inputs:
        inference_result: Inference result from inference engine

    Outputs:
        evaluation_result: Evaluation result based on the metric

    Note:
        This is an abstract class, child class must implement
        `load_evaluation_cfg` method.
    """

    SUPPORT_METRICS = {}

    def __init__(
        self,
        annotation_file: str,
        metric: Union[str, List[str]] = "AP",
        num_joints: int = 17,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.annotation_file = annotation_file
        self.num_joints = num_joints
        self.config = config if config else dict()
        self._metrics = set(metric) if isinstance(metric, list) else set([metric])
        for single_metric in self._metrics:
            if single_metric not in self.SUPPORT_METRICS:
                raise KeyError(f"metric {single_metric} is not supported")

        self._evaluation_cfg = self.load_evaluation_cfg()

        self.coco = self._load_ground_truth(self.annotation_file)
        self.id2name, self.name2id = self._get_mapping_id_name(self.coco.imgs)

        cat_ids = self.coco.getCatIds()
        cats = [cat["name"] for cat in self.coco.loadCats(cat_ids)]
        self.classes = ["__background__"] + cats
        self._class_to_coco_ind = dict(zip(cats, cat_ids))

    @property
    def metrics(self) -> Set[str]:
        """Returns the metrics used in evaluation."""
        return self._metrics

    def load_evaluation_cfg(self) -> Dict[str, Any]:
        """Loading the evaluation config, where the returned config must be a dictionary
        which stores the configuration of the engine, such as the using soft-nms, etc.

        Returns:
            Evaluation configurations
        """
        raise NotImplementedError("Child Class must implement this method.")

    def eval(self, inference_result: Dict[str, Any]) -> Dict[str, Any]:
        """Running the evaluation base on the inference result. Output the metric result.

        Args:
            inference_result: List of inference records

        Returns:
            metric result. Such as AP.5, etc.
        """
        raise NotImplementedError("Child Class must implement this method.")

    def __call__(self, inference_result: Dict[str, Any]) -> Dict[str, Any]:
        return self.eval(inference_result)

    def _load_ground_truth(self, annotation_file: str) -> COCO:
        coco = COCO(annotation_file=annotation_file)
        return coco

    def _write_coco_keypoint_results(
        self, keypoints: List[Dict[str, Any]], res_file: str
    ) -> None:
        """Write results into a json file."""
        data_pack = [
            {
                "cat_id": self._class_to_coco_ind[cls],
                "cls_ind": cls_ind,
                "cls": cls,
                "ann_type": "keypoints",
                "keypoints": keypoints,
            }
            for cls_ind, cls in enumerate(self.classes)
            if not cls == "__background__"
        ]

        results = self._coco_keypoint_results_one_category_kernel(data_pack[0])

        with open(res_file, "w") as f:
            json.dump(results, f, sort_keys=True, indent=4)

    def _coco_keypoint_results_one_category_kernel(
        self, data_pack: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Get coco keypoint results."""
        cat_id = data_pack["cat_id"]
        keypoints = data_pack["keypoints"]
        cat_results = []

        for img_kpts in keypoints:
            if not img_kpts:
                continue

            _key_points = np.array([img_kpt["keypoints"] for img_kpt in img_kpts])
            key_points = _key_points.reshape(-1, self.num_joints * 3)

            result = [
                {
                    "image_id": img_kpt["image_id"],
                    "category_id": cat_id,
                    "keypoints": key_point.tolist(),
                    "score": float(img_kpt["score"]),
                    "center": img_kpt.get("center", -1),
                    "scale": img_kpt.get("scale", -1),
                }
                for img_kpt, key_point in zip(img_kpts, key_points)
            ]

            cat_results.extend(result)

        return cat_results

    def _do_python_keypoint_eval(
        self, res_file: str
    ) -> List[Tuple[str, List[np.ndarray]]]:
        """Keypoint evaluation using COCOAPI."""
        coco_det = self.coco.loadRes(res_file)
        coco_eval = COCOeval(self.coco, coco_det, "keypoints")
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats_names = [
            "AP",
            "AP .5",
            "AP .75",
            "AP (M)",
            "AP (L)",
            "AR",
            "AR .5",
            "AR .75",
            "AR (M)",
            "AR (L)",
        ]

        info_str = list(zip(stats_names, coco_eval.stats))

        return info_str

    @staticmethod
    def _get_mapping_id_name(
        imgs: Dict[int, str]
    ) -> Tuple[Dict[int, str], Dict[str, int]]:
        id2name = {}
        name2id = {}
        for image_id, image in imgs.items():
            file_name = image["file_name"]
            id2name[image_id] = file_name
            name2id[file_name] = image_id

        return id2name, name2id

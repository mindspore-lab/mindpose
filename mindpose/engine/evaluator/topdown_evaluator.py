import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from xtcocotools.coco import COCO
from xtcocotools.cocoeval import COCOeval

from mindpose.utils.nms import oks_nms, soft_oks_nms
from ...register import register
from .evaluator import Evaluator


@register("evaluator", extra_name="topdown")
class TopDownEvaluator(Evaluator):
    """Create an evaluator based on Topdown method. It performs the model
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
        inference_result: Inference result from inference engine

    Outputs:
        evaluation_result: Evaluation result based on the metric
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
        self.coco = self.load_ground_truth(self.annotation_file)
        self.id2name, self.name2id = self._get_mapping_id_name(self.coco.imgs)

        cat_ids = self.coco.getCatIds()
        cats = [cat["name"] for cat in self.coco.loadCats(cat_ids)]
        self.classes = ["__background__"] + cats
        self._class_to_coco_ind = dict(zip(cats, cat_ids))

    def load_evaluation_cfg(self) -> Dict[str, Any]:
        """Loading the evaluation config, where the returned config must be a dictionary
        which stores the configuration of the engine, such as the using soft-nms, etc.

        Returns:
            Evaluation configurations
        """
        evaluation_cfg = dict()
        evaluation_cfg["vis_thr"] = self.config["vis_thr"]
        evaluation_cfg["oks_thr"] = self.config["oks_thr"]
        evaluation_cfg["use_nms"] = self.config["use_nms"]
        evaluation_cfg["soft_nms"] = self.config["soft_nms"]
        evaluation_cfg["sigmas"] = np.array(self.config["sigmas"])
        return evaluation_cfg

    def load_ground_truth(self, annotation_file: str) -> COCO:
        coco = COCO(annotation_file=annotation_file)
        return coco

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
            kpts[image_id].append(
                {
                    "keypoints": record["pred"],
                    "center": record["box"][0:2],
                    "scale": record["box"][2:4],
                    "area": record["box"][4],
                    "score": record["box"][5],
                    "image_id": image_id,
                    "bbox_id": record["bbox_id"],
                }
            )
        kpts = self._sort_and_unique_bboxes(kpts)

        # rescoring and oks nms
        vis_thr = self._evaluation_cfg["vis_thr"]
        oks_thr = self._evaluation_cfg["oks_thr"]
        valid_kpts = []
        for image_id in kpts.keys():
            img_kpts = kpts[image_id]
            for n_p in img_kpts:
                box_score = n_p["score"]
                kpt_score = 0
                valid_num = 0
                for n_jt in range(0, self.num_joints):
                    t_s = n_p["keypoints"][n_jt][2]
                    if t_s > vis_thr:
                        kpt_score = kpt_score + t_s
                        valid_num = valid_num + 1
                if valid_num != 0:
                    kpt_score = kpt_score / valid_num
                # rescoring
                n_p["score"] = kpt_score * box_score

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
                    "center": img_kpt["center"],
                    "scale": img_kpt["scale"],
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
        coco_eval = COCOeval(
            self.coco, coco_det, "keypoints", np.asarray(self._evaluation_cfg["sigmas"])
        )
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

    def _sort_and_unique_bboxes(self, kpts: Dict[int, Any], key: str = "bbox_id"):
        """sort kpts and remove the repeated ones."""
        for img_id, persons in kpts.items():
            num = len(persons)
            kpts[img_id] = sorted(kpts[img_id], key=lambda x: x[key])
            for i in range(num - 1, 0, -1):
                if kpts[img_id][i][key] == kpts[img_id][i - 1][key]:
                    del kpts[img_id][i]

        return kpts

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

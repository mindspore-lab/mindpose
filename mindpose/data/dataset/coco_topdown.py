import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
from xtcocotools.coco import COCO

from ...register import register

from .topdown import TopDownDataset


@register("dataset", extra_name="coco_topdown")
class COCOTopDownDataset(TopDownDataset):
    """Create an iterator for TopDown dataset based COCO annotation format
    return the tuple with image, center, scale, keypoints, rotation, target, target_weight for training,
    return the tuple with image, center, scale, rotation, image_file, box, box_id, box_score for evaluation

    Args:
        image_root: The path of the directory storing images
        annotation_file: The path of the annotation file
        is_train: Wether this dataset is used for training/testing
        use_gt_bbox_for_val: Use GT bbox instead of detection result during evaluation. Default: False
        detection_file: Path of the detection result. Defaul: None
        config: Method-specific configuration.

    Returns:
        image: Encoded data for image file
        center: Center (x, y) of the bounding box
        scale: Scale of the bounding box
        keypoints: Keypoints in (x, y, visibility)
        rotation: Rotatated degree
        target: A placeholder for later pipline using
        target_weight: A placeholder of later pipline using
        image_file: Path of the image file
        bbox: Bounding box coordinate (x, y, w, h)
        bbox_id: Bounding box id for each single image
        bbox_score: Bounding box score, 1 for ground truth
    """

    def load_dataset_cfg(self) -> Dict[str, Any]:
        """Loading the annoation info from the config file"""
        dataset_cfg = dict()
        dataset_cfg["image_size"] = np.array(self.config["image_size"])
        assert len(dataset_cfg["image_size"]) == 2
        dataset_cfg["det_bbox_thr"] = float(self.config["det_bbox_thr"])
        dataset_cfg["pixel_std"] = float(self.config["pixel_std"])
        dataset_cfg["scale_padding"] = float(self.config["scale_padding"])
        return dataset_cfg

    def load_dataset(self) -> List[Dict[str, Any]]:
        self.coco = COCO(self.annotation_file)
        self.id2name, self.name2id = self._get_mapping_id_name(self.coco.imgs)

        # load bbox and keypoints
        if self.is_train or self.use_gt_bbox_for_val:
            return self._load_coco_keypoint_annotations()

        # load bbox from detection
        return self._load_coco_detection_result()

    def _load_coco_keypoint_annotations(self) -> List[Dict[str, Any]]:
        """Ground truth bbox and keypoints."""
        # load image ids
        self.img_ids = self.coco.getImgIds()

        gt_db = []
        for img_id in self.img_ids:
            gt_db.extend(self._load_coco_keypoint_annotations_per_img(img_id))
        return gt_db

    def _load_coco_detection_result(self) -> List[Dict[str, Any]]:
        """Detection result"""
        with open(self.detection_file, "r") as f:
            all_boxes = json.load(f)

        bbox_id = 0
        kpt_db = []
        for det_res in all_boxes:
            if det_res["category_id"] != 1:
                continue

            image_file = os.path.join(
                self.image_root, self.id2name[det_res["image_id"]]
            )
            box = det_res["bbox"]
            score = det_res["score"]

            if score < self._dataset_cfg["det_bbox_thr"]:
                continue

            center, scale = self._xywh2cs(*box)
            kpt_db.append(
                {
                    "image_file": image_file,
                    "center": center,
                    "scale": scale,
                    "rotation": 0,
                    "boxes": box,
                    "bbox_ids": bbox_id,
                    "bbox_scores": score,
                }
            )
            bbox_id += 1
        return kpt_db

    def _load_coco_keypoint_annotations_per_img(
        self, img_id: int
    ) -> List[Dict[str, Any]]:
        img_ann = self.coco.loadImgs(img_id)[0]
        img_width = img_ann["width"]
        img_height = img_ann["height"]

        # no need to train crowd instance
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        annos = self.coco.loadAnns(ann_ids)

        annos = self._sanitize_bbox(annos, img_width, img_height)

        bbox_id = 0
        rec = []
        for anno in annos:
            if "keypoints" not in anno:
                continue
            if max(anno["keypoints"]) == 0:
                continue
            if "num_keypoints" in anno and anno["num_keypoints"] == 0:
                continue

            # keypoints store the info of x, y, visible for each joint
            keypoints = np.array(anno["keypoints"]).reshape(-1, 3)

            center, scale = self._xywh2cs(*anno["bbox"])

            image_file = os.path.join(self.image_root, self.id2name[img_id])
            rec.append(
                {
                    "image_file": image_file,
                    "center": center,
                    "scale": scale,
                    "keypoints": keypoints,
                    "rotation": 0,
                    "boxes": anno["bbox"],
                    "bbox_ids": bbox_id,
                    "bbox_scores": 1.0,
                }
            )
            bbox_id += 1

        return rec

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

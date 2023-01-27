import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
from xtcocotools.coco import COCO

from .topdown import TopDownDataset


class COCOTopDownDataset(TopDownDataset):
    """Create an iterator for COCO TopDown dataset."""

    def load_dataset_cfg(self) -> Dict[str, Any]:
        """Loading the annoation info from the config file"""
        dataset_cfg = dict()
        dataset_cfg["image_size"] = self.config.get("image_size", [192, 256])
        dataset_cfg["det_bbox_thr"] = self.config.get("det_bbox_thr", 0)
        dataset_cfg["pixel_std"] = float(self.config.get("pixel_std", 200.0))
        dataset_cfg["scale_padding"] = self.config.get("scale_padding", 1.25)
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

    def _load_coco_keypoint_annotations_per_img(self, img_id) -> List[Dict[str, Any]]:
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
        """
        Args:
            imgs (dict): dict of image info.

        Returns:
            tuple: Image name & id mapping dicts.

            - id2name (dict): Mapping image id to name.
            - name2id (dict): Mapping image name to id.
        """
        id2name = {}
        name2id = {}
        for image_id, image in imgs.items():
            file_name = image["file_name"]
            id2name[image_id] = file_name
            name2id[file_name] = image_id

        return id2name, name2id

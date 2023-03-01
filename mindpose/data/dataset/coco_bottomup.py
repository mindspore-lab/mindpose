import os
from typing import Any, Dict, List, Tuple

import numpy as np

import xtcocotools.mask
from xtcocotools.coco import COCO

from ...register import register

from .bottomup import BottomUpDataset


@register("dataset", extra_name="coco_bottomup")
class COCOBottomUpDataset(BottomUpDataset):
    """Create an iterator for ButtomUp dataset,
    return the tuple with (image, boxes, keypoints, mask, target, keypoint_coordinate)
    for training; return the tuple with (image, image_file) for evaluation

    Args:
        image_root: The path of the directory storing images
        annotation_file: The path of the annotation file. Default: None
        is_train: Wether this dataset is used for training/testing. Default: False
        num_joints: Number of joints in the dataset. Default: 17
        config: Method-specific configuration. Default: None

    Items in iterator:
        | image: Encoded data for image file
        | keypoints: Keypoints in (x, y, visibility)
        | mask: Mask of the image showing the valid annotations
        | target: A placeholder for later pipline using
        | keypoints_coordinate: A placeholder of later pipline using
        | image_file: Path of the image file
        | boxes: Bounding box coordinate (x0, y0), (x1, y1)
    """

    def load_dataset_cfg(self) -> Dict[str, Any]:
        """Loading the dataset config, where the returned config must be a dictionary
        which stores the configuration of the dataset, such as the image_size, etc.

        Returns:
            Dataset configurations
        """
        dataset_cfg = dict()
        return dataset_cfg

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Loading the dataset, where the returned record should contain the following key

        Keys:
            | image_file: Path of the image file
            | keypoints: Keypoints in (x, y, visibility)
            | boxes: Bounding box coordinate (x0, y0), (x1, y1)
            | mask: The mask of crowed or zero keypoints instances

        Returns:
            A list of records of groundtruth or predictions
        """
        self.coco = COCO(self.annotation_file)
        self.id2name, self.name2id = self._get_mapping_id_name(self.coco.imgs)

        # load bbox and keypoints
        return self._load_coco_keypoint_annotations()

    def _load_coco_keypoint_annotations(self) -> List[Dict[str, Any]]:
        """Ground truth bbox and keypoints."""
        # load image ids
        self.img_ids = self.coco.getImgIds()

        gt_db = []
        for img_id in self.img_ids:
            gt_db.append(self._load_coco_keypoint_annotations_per_img(img_id))
        return gt_db

    def _load_coco_keypoint_annotations_per_img(self, img_id: int) -> Dict[str, Any]:
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annos = self.coco.loadAnns(ann_ids)

        mask = self._get_mask(annos, img_id)
        annos = [
            obj for obj in annos if obj["iscrowd"] == 0 or obj["num_keypoints"] > 0
        ]
        keypoints = self._get_keypoints(annos)
        boxes = self._get_boxes(annos)

        image_file = os.path.join(self.image_root, self.id2name[img_id])

        rec = {
            "image_file": image_file,
            "keypoints": keypoints,
            "boxes": boxes,
            "mask": mask,
        }
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

    def _get_keypoints(self, annos: List[Dict[str, Any]]) -> np.ndarray:
        """Get the keypoints for single image"""
        if len(annos) == 0:
            return np.zeros((1, self.num_joints, 3))
        keypoints = [np.array(x["keypoints"]).reshape((-1, 3)) for x in annos]
        keypoints = np.stack(keypoints, axis=0)
        return keypoints

    def _get_boxes(self, annos: List[Dict[str, Any]]) -> np.ndarray:
        """Get the boxes for single image"""
        if len(annos) == 0:
            return np.zeros((1, 2, 2))
        boxes = [np.array(x["bbox"]) for x in annos]
        boxes = np.stack(boxes, axis=0)
        # xywh to xyxy
        boxes[..., 2] += boxes[..., 0]
        boxes[..., 3] += boxes[..., 1]
        boxes = boxes.reshape((-1, 2, 2))
        return boxes

    def _get_mask(self, annos: List[Dict[str, Any]], idx: int) -> np.ndarray:
        """Get ignore masks to mask out losses."""
        img_info = self.coco.loadImgs(idx)[0]

        m = np.zeros((img_info["height"], img_info["width"]), dtype=np.float32)

        for obj in annos:
            if "segmentation" in obj:
                if obj["iscrowd"]:
                    rle = xtcocotools.mask.frPyObjects(
                        obj["segmentation"], img_info["height"], img_info["width"]
                    )
                    m += xtcocotools.mask.decode(rle)
                elif obj["num_keypoints"] == 0:
                    rles = xtcocotools.mask.frPyObjects(
                        obj["segmentation"], img_info["height"], img_info["width"]
                    )
                    for rle in rles:
                        m += xtcocotools.mask.decode(rle)

        return m < 0.5

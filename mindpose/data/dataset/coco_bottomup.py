import os
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pycocotools.mask

from pycocotools.coco import COCO

from ...register import register

from .bottomup import BottomUpDataset


@register("dataset", extra_name="coco_bottomup")
class COCOBottomUpDataset(BottomUpDataset):
    """Create an iterator for ButtomUp dataset,
    return the tuple with (image, boxes, keypoints, mask, target, tag_ind)
    for training; return the tuple with (image, mask, center, scale, image_file,
    image_shape) for evaluation.

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
        dataset_cfg["sigma"] = float(self.config["sigma"])
        dataset_cfg["heatmap_sizes"] = self.config["heatmap_sizes"]
        dataset_cfg["expand_mask"] = self.config["expand_mask"]
        return dataset_cfg

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Loading the dataset, where the returned record should contain the following key

        Keys:
            | image_file: Path of the image file.
            | keypoints (For training only): Keypoints in (x, y, visibility).
            | boxes (For training only): Bounding box coordinate (x0, y0), (x1, y1).
            | mask_info (For training only): The mask info of crowed or zero keypoints
                instances.

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
            if self.is_train:
                # skip the images without annotations
                ann_ids = self.coco.getAnnIds(imgIds=img_id)
                if len(ann_ids) == 0:
                    continue
            gt_db.append(self._load_coco_keypoint_annotations_per_img(img_id))
        return gt_db

    def _load_coco_keypoint_annotations_per_img(self, img_id: int) -> Dict[str, Any]:
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annos = self.coco.loadAnns(ann_ids)

        mask_info = self._get_encoded_mask(annos, img_id)
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
            "mask_info": mask_info,
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

        # expand the keypoints by number of resolutions
        heatmap_sizes = self._dataset_cfg["heatmap_sizes"]
        num_levels = len(heatmap_sizes)
        keypoints = np.tile(keypoints[None, ...], (num_levels, 1, 1, 1))

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

    def _get_encoded_mask(
        self, annos: List[Dict[str, Any]], idx: int
    ) -> Dict[str, Any]:
        img_info = self.coco.loadImgs(idx)[0]

        height = img_info["height"]
        width = img_info["width"]

        m = np.zeros((height, width), dtype=np.float32)

        for obj in annos:
            if "segmentation" in obj:
                if obj["iscrowd"]:
                    rle = pycocotools.mask.frPyObjects(
                        obj["segmentation"], height, width
                    )
                    m += pycocotools.mask.decode(rle)
                elif obj["num_keypoints"] == 0:
                    rles = pycocotools.mask.frPyObjects(
                        obj["segmentation"], height, width
                    )
                    for rle in rles:
                        m += pycocotools.mask.decode(rle)

        m = m < 0.5

        # expand the mask by number of resolutions
        heatmap_sizes = self._dataset_cfg["heatmap_sizes"]
        num_levels = len(heatmap_sizes)
        m = np.tile(m[None, ...], (num_levels, 1, 1))

        if self._dataset_cfg["expand_mask"]:
            sigma = self._dataset_cfg["sigma"]
            for i in range(num_levels):
                # 3-sigma rule
                size = int(3 * sigma * (2 ** (num_levels - i)))
                kernel = np.zeros((2 * size + 1, 2 * size + 1), dtype=np.uint8)
                cv2.circle(kernel, (size, size), size, 1, -1)
                m[i] = cv2.erode(m[i].astype(np.uint8), kernel).astype(bool)

        encoded_m = np.packbits(m)

        mask_info = {"encoded_mask": encoded_m, "count": m.size, "shape": m.shape}
        return mask_info

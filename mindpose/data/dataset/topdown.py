from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class TopDownDataset:
    """Create an iterator for TopDown dataset,
    return the dict with key "img", "target" and "target_weight" for training,
    return the dict with key "img", "meta_data" for testing

    Args:
        image_root: The path of the directory storing images
        annotation_file: The path of the annotation file
        is_train: Wether this dataset is used for training/testing
        use_gt_bbox_for_val: Use GT bbox instead of detection result during evaluation. Default: False
        detection_file: Path of the detection result. Defaul: None
        config: Method-specific configuration.
    """

    def __init__(
        self,
        image_root: str,
        annotation_file: str,
        is_train: bool = False,
        use_gt_bbox_for_val: bool = False,
        detection_file: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.image_root = image_root
        self.annotation_file = annotation_file
        self.is_train = is_train
        self.use_gt_bbox_for_val = use_gt_bbox_for_val
        self.detection_file = detection_file
        self.config = config if config else dict()
        self._dataset_cfg = self.load_dataset_cfg()
        self._dataset = self.load_dataset()

        if self.annotation_file is None:
            if not self.is_train and not self.use_gt_bbox_for_val:
                raise ValueError(
                    "For evaluation, `detection_file` must be provided when `use_gt_bbox_for_val` is `False`"
                )

    def load_dataset_cfg(self) -> Dict[str, Any]:
        """Loading the dataset config"""
        raise NotImplementedError("Child class must implement this method.")

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Loading the dataset, where the returned record should containes the following key

        Keys:
            image_file: path of the image file
            center: center (x, y) of the bounding box
            scale: scale of the bounding box
            bbox: bounding box coordinate (x, y, w, h)
            keypoints: keypoints in (x, y, visibility)
            bbox_score: bounding box score, 1 for ground truth
            bbox_id: bounding box id for each single image
        """
        raise NotImplementedError("Child class must implement this method.")

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> List[np.ndarray]:
        record = self._dataset[idx]
        image = np.fromfile(record["image_file"], dtype=np.uint8)
        if self.is_train:
            return (
                image,
                record["center"],
                record["scale"],
                record["keypoints"],
                record["rotation"],
                np.float32(0),  # placeholder for target
                np.float32(0),  # placeholder for target_weight
            )
        return (
            image,
            record["center"],
            record["scale"],
            record["rotation"],
            record["image_file"],
            np.asarray(record["boxes"], dtype=np.float32),
            np.int32(record["bbox_ids"]),
            np.float32(record["bbox_scores"]),
        )

    @staticmethod
    def _sanitize_bbox(
        annos: List[Dict], img_width: int, img_height: int
    ) -> Dict[str, Any]:
        valid_annos = []
        for anno in annos:
            if "bbox" not in anno:
                continue
            x, y, w, h = anno["bbox"]
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(img_width - 1, x1 + max(0, w - 1))
            y2 = min(img_height - 1, y1 + max(0, h - 1))
            if ("area" not in anno or anno["area"] > 0) and x2 > x1 and y2 > y1:
                valid_anno = deepcopy(anno)
                valid_anno["bbox"] = [x1, y1, x2 - x1, y2 - y1]
                valid_annos.append(valid_anno)
        return valid_annos

    def _xywh2cs(
        self, x: float, y: float, w: float, h: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """This encodes bbox(x,y,w,h) into (center, scale)

        Args:
            x, y, w, h (float): left, top, width and height
            padding (float): bounding box padding factor

        Returns:
            center (np.ndarray[float32](2,)): center of the bbox (x, y).
            scale (np.ndarray[float32](2,)): scale of the bbox w & h.
        """
        aspect_ratio = (
            self._dataset_cfg["image_size"][0] / self._dataset_cfg["image_size"][1]
        )
        center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)

        if self.is_train and np.random.rand() < 0.3:
            center += 0.4 * (np.random.rand(2) - 0.5) * [w, h]

        if w > aspect_ratio * h:
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio

        pixel_std = self._dataset_cfg["pixel_std"]
        scale_padding = self._dataset_cfg["scale_padding"]

        scale = np.array([w / pixel_std, h / pixel_std], dtype=np.float32)
        # padding to include proper amount of context
        scale = scale * scale_padding
        return center, scale

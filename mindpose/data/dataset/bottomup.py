import logging
from typing import Any, Dict, List, Optional

import numpy as np


class BottomUpDataset:
    """Create an iterator for ButtomUp dataset,
    return the tuple with (image, boxes, keypoints, target, mask, tag_ind)
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
        | tag_ind: A placeholder of later pipline using
        | image_file: Path of the image file
        | boxes: Bounding box coordinate (x0, y0), (x1, y1)

    Note:
        This is an abstract class, child class must implement
        `load_dataset_cfg` and `load_dataset` method.
    """

    def __init__(
        self,
        image_root: str,
        annotation_file: Optional[str] = None,
        is_train: bool = False,
        num_joints: int = 17,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.image_root = image_root
        self.annotation_file = annotation_file
        self.is_train = is_train
        self.num_joints = num_joints
        self.config = config if config else dict()
        self._dataset_cfg = self.load_dataset_cfg()
        self._dataset = self.load_dataset()
        logging.info(f"Number of records in dataset: {len(self._dataset)}")

    def load_dataset_cfg(self) -> Dict[str, Any]:
        """Loading the dataset config, where the returned config must be a dictionary
        which stores the configuration of the dataset, such as the image_size, etc.

        Returns:
            Dataset configurations
        """
        raise NotImplementedError("Child class must implement this method.")

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
        raise NotImplementedError("Child class must implement this method.")

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> List[np.ndarray]:
        record = self._dataset[idx]
        image = np.fromfile(record["image_file"], dtype=np.uint8)
        if self.is_train:
            # decode mask
            mask_info = record["mask_info"]
            mask = np.unpackbits(
                mask_info["encoded_mask"], count=mask_info["count"]
            ).reshape(mask_info["shape"])
            return (
                image,
                np.asarray(record["boxes"], dtype=np.float32),
                np.asarray(record["keypoints"], dtype=np.float32),
                np.float32(0),  # placeholder for target
                np.asarray(mask, dtype=np.uint8),
                np.int32(0),  # placeholder for tag_ind
            )
        return (
            image,
            np.uint8(0),  # placeholder for mask
            np.float32(0),  # placeholder for center
            np.float32(0),  # placeholder for scale
            record["image_file"],
            np.int32(0),  # placeholder for image_shape
        )

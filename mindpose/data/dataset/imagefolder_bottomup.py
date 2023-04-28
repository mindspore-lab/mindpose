import os
from typing import Any, Dict, List

from ...register import register

from .bottomup import BottomUpDataset


@register("dataset", extra_name="imagefolder_bottomup")
class ImageFolderBottomUpDataset(BottomUpDataset):
    """Create an iterator for ButtomUp dataset based on image folder.
    It is usually used for demo usage. Return the tuple with (image,
    mask, center, scale, image_file, image_shape)

    Args:
        image_root: The path of the directory storing images
        annotation_file: The path of the annotation file. Default: None
        is_train: Wether this dataset is used for training/testing. Default: False
        num_joints: Number of joints in the dataset. Default: 17
        config: Method-specific configuration. Default: None
    """

    SUPPORTED_EXTS = {".bmp", ".png", ".jpg", ".jpeg", ".tiff"}

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
            | image_file: Path of the image file.

        Returns:
            A list of records of groundtruth or predictions
        """
        image_files = self._search_images(self.image_root)
        records = []
        for image_file in image_files:
            records.append({"image_file": image_file})
        return records

    def _search_images(self, image_root) -> List[str]:
        files = os.listdir(image_root)
        files = [
            x for x in files if os.path.splitext(x)[1].lower() in self.SUPPORTED_EXTS
        ]
        files = [os.path.join(image_root, x) for x in files]
        return files

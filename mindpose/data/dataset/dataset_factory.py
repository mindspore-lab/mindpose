from typing import Any, Dict, Optional

from mindspore.dataset import GeneratorDataset

from ..column_names import COLUMN_MAP

from .coco_topdown import COCOTopDownDataset

_SUPPORT_DATASET = {"coco_topdown": COCOTopDownDataset}


def create_dataset(
    image_root: str,
    annotation_file: str,
    dataset_format: str = "coco_topdown",
    is_train: bool = True,
    use_gt_bbox_for_val: bool = False,
    detection_file: Optional[str] = None,
    device_num: Optional[int] = None,
    rank_id: Optional[int] = None,
    num_workers: int = 1,
    config: Optional[Dict[str, Any]] = None,
) -> GeneratorDataset:
    """Create dataset.

    Args:
        image_root: The path of the directory storing images
        annotation_file: The path of the annotation file
        dataset_format: The dataset format. Different format yield different final output.
            Currenty it supports `coco_topdown` Default: `coco_topdown`
        is_train: Wether this dataset is used for training/testing
        use_gt_bbox_for_val: Use GT bbox instead of detection result during evaluation. Default: False
        detection_file: Path of the detection result. Defaul: None
        device_num: Number of devices (e.g. GPU). Default: None
        rank_id: Current process's rank id. Default: None
        num_workers: Number of workers in reading data. Default: 1
        config: Method-specific configuration.

    Returns:
        GeneratorDataset Object
    """
    try:
        dataset = _SUPPORT_DATASET[dataset_format](
            image_root,
            annotation_file,
            is_train=is_train,
            use_gt_bbox_for_val=use_gt_bbox_for_val,
            detection_file=detection_file,
            config=config,
        )
    except KeyError as e:
        raise ValueError(
            "Unsupported dataset format. "
            f"Currently it only support {_SUPPORT_DATASET.keys()}"
        ) from e

    # select the column names for different task
    if is_train:
        column_names = COLUMN_MAP[dataset_format]["train"]
    else:
        column_names = COLUMN_MAP[dataset_format]["val"]

    dataset = GeneratorDataset(
        dataset,
        column_names=column_names,
        shuffle=is_train,
        num_shards=device_num,
        shard_id=rank_id,
        num_parallel_workers=num_workers,
    )

    return dataset

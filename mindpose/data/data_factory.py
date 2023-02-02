from typing import Any, Dict, List, Optional, Union

import mindspore.dataset.vision as vision
import numpy as np
from mindspore.dataset import Dataset, GeneratorDataset

from ..register import entrypoint
from .column_names import COLUMN_MAP, FINAL_COLUMN_MAP

from .transform import Transform


__all__ = ["create_dataset", "create_pipeline"]


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
    """Create dataset for training or evaluation.

    Args:
        image_root: The path of the directory storing images
        annotation_file: The path of the annotation file
        dataset_format: The dataset format. Different format yield
            different final output. Default: `coco_topdown`
        is_train: Wether this dataset is used for training/testing: Default: True
        use_gt_bbox_for_val: Use GT bbox instead of detection result
            during evaluation. Default: False
        detection_file: Path of the detection result. Default: None
        device_num: Number of devices (e.g. GPU). Default: None
        rank_id: Current process's rank id. Default: None
        num_workers: Number of workers in reading data. Default: 1
        config: Dataset-specific configuration

    Returns:
        Dataset for training or evaluation
    """
    dataset = entrypoint("dataset", dataset_format)(
        image_root,
        annotation_file,
        is_train=is_train,
        use_gt_bbox_for_val=use_gt_bbox_for_val,
        detection_file=detection_file,
        config=config,
    )

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


def create_pipeline(
    dataset: Dataset,
    transforms: List[Union[str, Dict[str, Any]]],
    method: str = "topdown",
    batch_size: int = 1,
    is_train: bool = True,
    normalize: bool = True,
    normalize_mean: List[float] = [0.485, 0.456, 0.406],
    normalize_std: List[float] = [0.229, 0.224, 0.255],
    hwc_to_chw: bool = True,
    num_workers: int = 1,
    config: Optional[Dict[str, Any]] = None,
) -> Dataset:
    """Create dataset tranform pipeline. The returned datatset is transformed
    sequentially based on the given list of transforms.

    Args:
        dataset: Dataset to perform transformations
        transforms: List of transformations
        method: The method to use. Default: "topdown"
        batch_size: Batch size. Default: 1
        is_train: Whether the transformation is for training/testing. Default: True
        normalize: Perform normalization. Default: True
        normalize_mean: Mean of the normalization: Default: [0.485, 0.456, 0.406]
        normalize_std: Std of the normalization: Default: [0.229, 0.224, 0.255]
        hwc_to_chw: Wwap height x width x channel to
            channel x height x width. Default: True
        num_workers: Number of workers in processing data. Default: 1
        config: Transform-specific configuration

    Returns:
        The transformed dataset
    """
    if is_train:
        column_names = COLUMN_MAP[method]["train"]
        final_column_names = FINAL_COLUMN_MAP[method]["train"]
    else:
        column_names = COLUMN_MAP[method]["val"]
        final_column_names = FINAL_COLUMN_MAP[method]["val"]

    # prepare transformations
    transform_funcs = _convert_names_to_transform(
        transforms, is_train=is_train, config=config
    )

    # decode
    dataset = dataset.map(
        vision.Decode(), input_columns=["image"], num_parallel_workers=num_workers
    )

    # perform transformation on data and label
    dataset = dataset.map(
        transform_funcs, input_columns=column_names, num_parallel_workers=num_workers
    )

    # image normalization and swap channel
    image_funcs = []
    if normalize:
        mean = np.array(normalize_mean) * 255.0
        std = np.array(normalize_std) * 255.0
        image_funcs.append(vision.Normalize(mean.tolist(), std.tolist()))
    if hwc_to_chw:
        image_funcs.append(vision.HWC2CHW())

    if image_funcs:
        dataset = dataset.map(
            image_funcs, input_columns=["image"], num_parallel_workers=num_workers
        )

    # remove unessary outputs
    dataset = dataset.project(final_column_names)

    # batch
    dataset = dataset.batch(batch_size, drop_remainder=is_train)
    return dataset


def _convert_names_to_transform(
    names_with_args: List[Union[str, Dict[str, Any]]],
    is_train: bool = True,
    config: Optional[Dict[str, Any]] = None,
) -> List[Transform]:
    """Convert a list of names with arguments to a list of transforms funcs."""
    transforms = list()
    for name_with_arg in names_with_args:
        if isinstance(name_with_arg, str):
            name, kwargs = name_with_arg, dict()
        else:
            name = list(name_with_arg.keys())[0]
            # combine list of dict into single dict
            # it will be nice if there is a better approach
            kwargs = list(name_with_arg.values())[0]
            kwargs = {list(x.keys())[0]: list(x.values())[0] for x in kwargs}

        transform = entrypoint("transform", name)(
            is_train=is_train, config=config, **kwargs
        )
        transforms.append(transform)
    return transforms

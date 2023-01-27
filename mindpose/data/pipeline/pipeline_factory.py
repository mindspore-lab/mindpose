from typing import Any, Dict, List, Optional, Union

import mindspore.dataset.vision as vision
from mindspore.dataset import Dataset

from ..column_names import COLUMN_MAP, FINAL_COLUMN_MAP
from .topdown_transform import TOPDOWN_TRANSFORM_MAPPING
from .transform import Transform

# merge all available transforms
TRANSFORM_MAPPING = TOPDOWN_TRANSFORM_MAPPING


def create_pipeline(
    dataset: Dataset,
    transforms: List[Union[str, Dict[str, Any]]],
    method: str = "topdown",
    batch_size: int = 1,
    is_train: bool = False,
    normalize: bool = True,
    num_joints: int = 17,
    config: Optional[Dict[str, Any]] = None,
) -> Dataset:
    """Create dataset tranform pipeline

    Args:
        dataset: Dataset to perform transformations
        transforms: List of transformations
        method: The method to use. Default: "Topdown"
        batch_size: batch size. Default: 1
        is_train: whether the transformation is for training/testing,
            since some of the transform behaves different. Default: False
        normalize: perform normalization
            and swap height x width x channel to channel x height x width. Default: True
        num_joints: Number of joints. Default: 17.
        config: Method-specific configuration.
    """
    if is_train:
        column_names = COLUMN_MAP[method]["train"]
        final_column_names = FINAL_COLUMN_MAP[method]["train"]
    else:
        column_names = COLUMN_MAP[method]["val"]
        final_column_names = FINAL_COLUMN_MAP[method]["val"]

    # prepare transformations
    transform_funcs = _convert_names_to_transform(
        transforms, is_train=is_train, num_joints=num_joints, config=config
    )

    # decode
    dataset = dataset.map(vision.Decode(), input_columns=["image"])

    # perform transformation on data and label
    dataset = dataset.map(transform_funcs, input_columns=column_names)

    # image normalization
    if normalize:
        dataset = dataset.map(
            [
                vision.Normalize(
                    mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                    std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
                ),
                vision.HWC2CHW(),
            ],
            input_columns=["image"],
        )

    # remove unessary outputs
    dataset = dataset.project(final_column_names)

    # batch
    dataset = dataset.batch(batch_size, drop_remainder=is_train)
    return dataset


def _convert_names_to_transform(
    names_with_args: List[Union[str, Dict[str, Any]]],
    is_train: bool = False,
    num_joints: int = 17,
    config: Optional[Dict[str, Any]] = None,
) -> List[Transform]:
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
        try:
            transform = TRANSFORM_MAPPING[name](
                is_train=is_train, num_joints=num_joints, config=config, **kwargs
            )
            transforms.append(transform)
        except KeyError as e:
            raise ValueError(
                f"Unsupport transformation `{name}`, "
                f"Available transformation: `{TRANSFORM_MAPPING.keys()}`"
            ) from e
    return transforms

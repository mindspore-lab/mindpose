import logging
from typing import Any, Dict, List, Optional, Union

from ..models.networks import EvalNet
from ..register import entrypoint
from .evaluator import Evaluator
from .inferencer import Inferencer


def create_inferencer(
    net: EvalNet,
    name: str = "topdown_heatmap",
    config: Optional[Dict[str, Any]] = None,
    dataset_config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Inferencer:
    """Create inference engine

    Args:
        net: Network for evaluation
        name: Name of the inference method. Default: "topdown_heatmap"
        config: Inference config. Default: None
        dataset_config: Dataset config. Since the inference method sometimes relies on the dataset info. Default: None
        **kwargs: Arguments which feed into the inferencer
    """
    config = config if config else dict()
    dataset_config = dataset_config if dataset_config else dict()

    # combine two configurations
    full_config = merge_configs(config, dataset_config)

    return entrypoint("inferencer", name)(net=net, config=full_config, **kwargs)


def create_evaluator(
    annotation_file: str,
    name: str = "topdown",
    metric: Union[str, List[str]] = "AP",
    config: Optional[Dict[str, Any]] = None,
    dataset_config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Evaluator:
    """Create evaluator

    Args:
        name: Name of the evaluation method. Default: "topdown"
        annotation_file: Path of the annotation file. It only supports COCO-format.
        metric: Supported metrics. Default: "AP"
        config: Evaluaton config. Default: None
        dataset_config: Dataset config. Since the evaluation method sometimes relies on the dataset info. Default: None
        **kwargs: Arguments which feed into the evaluator
    """
    config = config if config else dict()
    dataset_config = dataset_config if dataset_config else dict()

    # combine two configurations
    full_config = merge_configs(config, dataset_config)

    return entrypoint("evaluator", name)(
        annotation_file=annotation_file, metric=metric, config=full_config, **kwargs
    )


def merge_configs(config_1: Dict[str, Any], config_2: Dict[str, Any]) -> Dict[str, Any]:
    common_keys = set(config_1.keys()).intersection(config_2.keys())
    if len(common_keys) > 0:
        logging.warning(f"Duplicated keys found in two configs: `{common_keys}`")
    merged_config = {**config_1, **config_2}
    return merged_config

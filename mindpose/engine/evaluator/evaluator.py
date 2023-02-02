from typing import Any, Dict, List, Optional, Set, Union


class Evaluator:
    """Create an evaluator engine.
    This is an abstract class, child class must implement `load_evaluation_cfg` and `__call__` method.

    Args:
        annotation_file: Path of the annotation file. It only supports COCO-format.
        metric: Supported metrics. Default: "AP"
        num_joints: Number of joints. Default: 17
        config: Method-specific configuration. Default: None

    Inputs:
        inference_result: Inference result from inference engine

    Outputs:
        evaluation_result: Evaluation result based on the metric
    """

    SUPPORT_METRICS = {}

    def __init__(
        self,
        annotation_file: str,
        metric: Union[str, List[str]] = "AP",
        num_joints: int = 17,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.annotation_file = annotation_file
        self.num_joints = num_joints
        self.config = config if config else dict()
        self._metrics = set(metric) if isinstance(metric, list) else set([metric])
        for single_metric in self._metrics:
            if single_metric not in self.SUPPORT_METRICS:
                raise KeyError(f"metric {single_metric} is not supported")

        self._evaluation_cfg = self.load_evaluation_cfg()

    @property
    def metrics(self) -> Set[str]:
        return self._metrics

    def load_evaluation_cfg(self) -> Dict[str, Any]:
        """Loading the evaluation config, where the returned config must be a dictionary
        which stores the configuration of the engine, such as the using TTA, etc.
        """
        raise NotImplementedError("Child Class must implement this method.")

    def __call__(self, inference_result: Dict[str, Any]) -> Dict[str, Any]:
        """Running the evaluation on the inference result"""
        raise NotImplementedError("Child Class must implement this method.")

from typing import Any, Dict, List, Optional, Set, Union


class Evaluator:
    SUPPORT_METRICS = {}

    def __init__(
        self,
        annotation_file: str,
        metric: Union[str, List[str]] = "AP",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.annotation_file = annotation_file
        self.config = config if config else dict()
        self.metrics = set(metric) if isinstance(metric, list) else set([metric])
        for single_metric in self.metrics:
            if single_metric not in self.SUPPORT_METRICS:
                raise KeyError(f"metric {single_metric} is not supported")

        self._evaluation_cfg = self.load_evaluation_cfg()

    def get_metrics(self) -> Set[str]:
        return self.metrics

    def load_evaluation_cfg(self) -> Dict[str, Any]:
        raise NotImplementedError("Child Class must implement this method.")

    def evaluate(*args: Any, **kwargs: Any) -> Dict[str, float]:
        raise NotImplementedError("Child Class must implement this method.")

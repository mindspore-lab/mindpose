from typing import Any, Dict, Optional

from .evaluator import Evaluator
from .topdown_evaluator import TopDownEvaluator


def create_evaluator(
    name: str = "topdown", config: Optional[Dict[str, Any]] = None, **kwargs: Any
) -> Evaluator:
    if name == "topdown":
        return TopDownEvaluator(config=config, **kwargs)
    else:
        raise ValueError("Unsupported Evaluator")

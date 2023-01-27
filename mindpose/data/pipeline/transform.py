from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class Transform:
    """Create transform

    Args:
        is_train: whether the transformation is for training/testing,
            since some of the transform behaves different. Default: False
        num_joints: Number of joints. Default: 17.
        config: Method-specific configuration.

    Inputs:
        data: Data tuples

    Outputs:
        result: Transformed data tuples
    """

    def __init__(
        self,
        is_train: bool = False,
        num_joints: int = 17,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.is_train = is_train
        self.num_joints = num_joints
        self.config = config if config else dict()
        self._transform_cfg = self.load_transform_cfg()
        self._required_field = self.setup_required_field()

    def load_transform_cfg(self) -> Dict[str, Any]:
        raise NotImplementedError("Child class must implement this method.")

    def transform(self, state: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Child class must implement this method.")

    def setup_required_field(self) -> List[str]:
        raise NotImplementedError("Child class must implement this method.")

    def __call__(self, *args: Any) -> Tuple[np.ndarray, ...]:
        # pack the arguments
        states = dict(zip(self._required_field, args))
        transformed_states = self.transform(states)
        states.update(transformed_states)

        # unpack the argument for mindspore dataset API
        final_states = {k: states[k] for k in self._required_field}
        return tuple(final_states.values())

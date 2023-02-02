from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class Transform:
    """Transform the input data into the output data.

    Args:
        is_train: Whether the transformation is for training/testing. Default: True
        config: Method-specific configuration. Default: None

    Inputs:
        data: Data tuples need to be transformed

    Outputs:
        result: Transformed data tuples

    Note:
        This is an abstract class, child class must implement `load_transform_cfg`,
        `transform` and `setup_required_field` method.
    """

    def __init__(
        self,
        is_train: bool = True,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.is_train = is_train
        self.config = config if config else dict()
        self._transform_cfg = self.load_transform_cfg()
        self._required_field = self.setup_required_field()

    def load_transform_cfg(self) -> Dict[str, Any]:
        """Loading the transform config, where the returned the config must
        be a dictionary which stores the configuration of this transformation,
        such as the transformed image size, etc.

        Returns:
            Transform configuration
        """
        raise NotImplementedError("Child class must implement this method.")

    def transform(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Transform the state into the transformed state. state is a dictionay
        storing the informaton of the image and labels, the returned states is
        the updated dictionary storing the updated image and labels.

        Args:
            state: Stored information of image and labels

        Returns:
            Updated inforamtion of image and labels based on the transformation
        """
        raise NotImplementedError("Child class must implement this method.")

    def setup_required_field(self) -> List[str]:
        """Get the required columns names used for this transformation.
        The columns names will be later used with Minspore Dataset `map` func.

        Returns:
            The column names
        """
        raise NotImplementedError("Child class must implement this method.")

    def __call__(self, *args: Any) -> Tuple[np.ndarray, ...]:
        """This simply does the following process
        1. Pack the column names and data tuples into a dictionary
        2. Calling the tranform method on the dictionary
        3. Unpack the dictionay and return the data tuples only"""
        # pack the arguments
        states = dict(zip(self._required_field, args))
        transformed_states = self.transform(states)
        states.update(transformed_states)

        # unpack the argument for mindspore dataset API
        final_states = {k: states[k] for k in self._required_field}
        return tuple(final_states.values())

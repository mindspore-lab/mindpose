from typing import List, Union

import mindspore.nn as nn


class Neck(nn.Cell):
    """Abstract class for all necks.
    Child class must implement `construct` and `out_channels` method.
    """

    @property
    def out_channels(self) -> Union[List[int], int]:
        """Get number of output channels.

        Returns:
            Output channels.
        """
        raise NotImplementedError("Child class must implement this method.")

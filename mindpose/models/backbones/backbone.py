from typing import List, Union

import mindspore.nn as nn
from mindspore import Tensor


class Backbone(nn.Cell):
    """Abstract class for all backbones.

    Note:
        Child class must implement `foward_feature` and `out_channels` method.
    """

    def forward_feature(self, x: Tensor) -> Tensor:
        """Perform the feature extraction.

        Args:
            x: Tensor

        Returns:
            Extracted feature
        """
        raise NotImplementedError("Child class must implement this method.")

    def construct(self, x: Tensor) -> Tensor:
        x = self.forward_feature(x)
        return x

    @property
    def out_channels(self) -> Union[List[int], int]:
        """Get number of output channels.

        Returns:
            Output channels.
        """
        raise NotImplementedError("Child class must implement this method.")

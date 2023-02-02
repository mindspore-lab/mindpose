import mindspore.nn as nn
from mindspore import Tensor


class Backbone(nn.Cell):
    """Abstract class for all backbones"""

    def forward_feature(self, x: Tensor) -> Tensor:
        """Perform the feature extraction.

        Args:
            x: Tensor

        Returns:
            Tensor: Extracted feature
        """
        raise NotImplementedError("Child class must implement this method.")

    def construct(self, x: Tensor) -> Tensor:
        x = self.forward_feature(x)
        return x

    @property
    def out_channels(self) -> int:
        """Get number of output channels"""
        raise NotImplementedError("Child class must implement this method.")

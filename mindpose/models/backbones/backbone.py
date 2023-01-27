import mindspore.nn as nn


class Backbone(nn.Cell):
    """Abstract class for all backbones"""

    @property
    def out_channels(self) -> int:
        raise NotImplementedError("Child class must implement this method.")

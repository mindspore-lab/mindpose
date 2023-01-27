import mindspore.nn as nn


class Neck(nn.Cell):
    """Abstract class for all neck"""

    @property
    def out_channels(self) -> int:
        raise NotImplementedError("Child class must implement this method.")

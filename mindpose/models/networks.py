"""Defines all necessary networks for training / evaluation
"""
from typing import Optional, Tuple

import mindspore.nn as nn
from mindspore import Tensor

from .backbones import Backbone
from .decoders import Decoder
from .heads import Head
from .loss import Loss
from .necks import Neck


class Net(nn.Cell):
    """Create network for foward and backward propagate.

    Args:
        backbone: Model backbone
        head: Model head
        neck: Model neck. Default: None

    Inputs:
        | x: Tensor

    Outputs:
        | result: Tensor
    """

    def __init__(
        self, backbone: Backbone, head: Head, neck: Optional[Neck] = None
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.neck = neck
        self.has_neck = self.neck is not None

    def construct(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        if self.has_neck:
            x = self.neck(x)
        x = self.head(x)
        return x


class EvalNet(nn.Cell):
    """Create network for forward propagate and decoding only.

    Args:
        net: Network used for foward and backward propagate
        decoder: Decoder
        output_raw: Return extra net's ouput. Default: True

    Inputs:
        | inputs: List of tensors

    Outputs
        | result: Decoded result
        | raw_result (optional): Raw result if output_raw is true
    """

    def __init__(self, net: Net, decoder: Decoder, output_raw: bool = True) -> None:
        super().__init__()
        self.net = net
        self.decoder = decoder
        self.output_raw = output_raw
        self.net.set_train(False)
        self.decoder.set_train(False)

    def construct(self, *inputs: Tensor) -> Tuple[Tensor, ...]:
        x = self.net(inputs[0])
        result = self.decoder(x, *inputs[1:])
        if self.output_raw:
            return result, x
        return result


class NetWithLoss(nn.Cell):
    """Create network with loss.

    Args:
        net: Network used for foward and backward propagate
        loss: Loss cell
        has_extra_inputs: Has Extra inputs in the loss calculation. Default: False

    Inputs:
        | data: Tensor feed into network
        | label: Tensor of label
        | extra_inputs: List of extra tensors used in loss calculation

    Outputs:
        | loss: Loss value
    """

    def __init__(self, net: Net, loss: Loss, has_extra_inputs: bool = False) -> None:
        super().__init__()
        self.net = net
        self.loss = loss
        self.has_extra_inputs = has_extra_inputs

    def construct(self, data: Tensor, label: Tensor, *extra_inputs: Tensor) -> Tensor:
        out = self.net(data)
        if self.has_extra_inputs:
            return self.loss(out, label, *extra_inputs)
        return self.loss(out, label)

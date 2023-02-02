from typing import Any, Dict, Optional

from ..register import entrypoint
from .backbones import Backbone
from .decoders import Decoder
from .heads import Head
from .loss import Loss
from .necks import Neck
from .networks import EvalNet, Net, NetWithLoss


__all__ = [
    "create_backbone",
    "create_neck",
    "create_head",
    "create_decoder",
    "create_loss",
    "create_network",
    "create_eval_network",
    "create_network_with_loss",
]


def create_backbone(
    name: str,
    pretrained: bool = False,
    ckpt_url: str = "",
    in_channels: int = 3,
    **kwargs: Any,
) -> Backbone:
    """Create model backbone.

    Args:
        name: Name of the backbone
        pretrained: Whether the backbone is pretrained. Default: False
        ckpt_url: Url of the pretrain check point. Default: None
        in_channels: Number of channels in the input data. Default: 3
        **kwargs: Arguments which feed into the backbone

    Returns:
        Model backbone
    """
    return entrypoint("backbone", name)(
        pretrained=pretrained, ckpt_url=ckpt_url, in_channels=in_channels, **kwargs
    )


def create_head(name: str, in_channels, num_joints: int = 17, **kwargs: Any) -> Head:
    """Create model head.

    Args:
        name: Name of the head
        in_channels: Number of channels in the input tensor
        num_joints: Number of joints. Default: 17
        **kwargs: Arguments which feed into the head

    Returns:
        Model head
    """
    return entrypoint("head", name)(
        in_channels=in_channels, num_joints=num_joints, **kwargs
    )


def create_neck(name: str, in_channels, out_channels, **kwargs: Any) -> Neck:
    """Create model neck.

    Args:
        name: Name of the neck
        in_channels: Number of channels in the input tensor
        out_channels: Number of channels in the output tensor
        **kwargs: Arguments which feed into the neck

    Returns:
        Model neck
    """
    return entrypoint("neck", name)(
        in_channels=in_channels, out_channels=out_channels, **kwargs
    )


def create_decoder(name: str, **kwargs: Any) -> Decoder:
    """Create model decoder.

    Args:
        name: Name of the decoder
        **kwargs: Arguments which feed into the decoder

    Returns:
        Model decoder
    """
    return entrypoint("decoder", name)(**kwargs)


def create_loss(name: str, **kwargs: Any) -> Loss:
    """Create model loss.

    Args:
        name: Name of the loss
        **kwargs: Arguments which feed into the loss

    Returns:
        Loss
    """
    return entrypoint("loss", name)(**kwargs)


def create_network(
    backbone_name: str,
    head_name: str,
    neck_name: str = "",
    backbone_pretrained: bool = False,
    backbone_ckpt_url: str = "",
    in_channels: int = 3,
    neck_out_channels: int = 256,
    num_joints: int = 17,
    backbone_args: Optional[Dict[str, Any]] = None,
    neck_args: Optional[Dict[str, Any]] = None,
    head_args: Optional[Dict[str, Any]] = None,
) -> Net:
    """Create network for training.

    Args:
        backbone_name: Backbone name
        head_name: Head name
        neck_name: Neck name. Default: ""
        backbone_pretrained: Whether backbone is pretrained. Default: False
        backbone_ckpt_url: Url of backbone's pretrained checkpoint. Default: ""
        in_channels: Number of channels in the input data. Default: 3
        neck_out_channels: Number of output channels in the neck. Default: 256
        num_joints: Number of joints in the output. Default: 17
        backbone_args: Arguments for backbone. Defauult: None
        neck_args: Arguments for neck. Default: None
        head_args: Arguments for head: Default: None

    Returns:
        Network for training
    """
    backbone_args = backbone_args if backbone_args else dict()
    neck_args = neck_args if neck_args else dict()
    head_args = head_args if head_args else dict()

    backbone = create_backbone(
        backbone_name,
        pretrained=backbone_pretrained,
        ckpt_url=backbone_ckpt_url,
        in_channels=in_channels,
        **backbone_args,
    )

    if neck_name:
        neck = create_neck(
            neck_name,
            in_channels=backbone.out_channels,
            out_channels=neck_out_channels,
            **neck_args,
        )
        head = create_head(
            head_name, in_channels=neck.out_channels, num_joints=num_joints, **head_args
        )
    else:
        neck = None
        head = create_head(
            head_name,
            in_channels=backbone.out_channels,
            num_joints=num_joints,
            **head_args,
        )

    net = Net(backbone, head, neck=neck)
    return net


def create_eval_network(net: Net, decoder: Decoder, output_raw: bool = True) -> EvalNet:
    """Create network for inferencing or evaluation.

    Args:
        net: Network used for foward and backward propagate
        decoder: Decoder
        output_raw: Return extra net's ouput. Default: True

    Returns:
        Network for inferencing or evaluation
    """
    net = EvalNet(net, decoder, output_raw=output_raw)
    return net


def create_network_with_loss(
    net: Net, loss: Loss, has_extra_inputs: bool = False
) -> NetWithLoss:
    """Create network with loss for training.

    Args:
        net: Network used for foward and backward propagate
        loss: Loss cell
        has_extra_inputs: Has Extra inputs in the loss calculation. Default: False

    Returns:
        Network with loss for training
    """
    net = NetWithLoss(net, loss, has_extra_inputs=has_extra_inputs)
    return net

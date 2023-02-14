from typing import Any, Dict, List, Union

from mindcv.optim.adamw import AdamW

from mindspore.nn.learning_rate_schedule import LearningRateSchedule
from mindspore.nn.optim import Adagrad, Adam, Momentum, Optimizer, SGD

from ..register import entrypoint, register


# register the default optimizers
Adam = register("optim", extra_name="adam")(Adam)
SGD = register("optim", extra_name="sgd")(SGD)
Momentum = register("optim", extra_name="momentum")(Momentum)
Adagrad = register("optim", extra_name="adagrad")(Adagrad)
AdamW = register("optim", extra_name="adamw")(AdamW)


def init_group_params(
    params: List[Any], weight_decay: float = 0
) -> List[Dict[str, Any]]:
    """Split the parameters into groups. Parameters for BN and bias has no decay."""
    decay_params = []
    no_decay_params = []

    for param in params:
        if (
            "beta" not in param.name
            and "gamma" not in param.name
            and "bias" not in param.name
        ):
            decay_params.append(param)
        else:
            no_decay_params.append(param)
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params},
        {"order_params": params},
    ]


def create_optimizer(
    params: List[Any],
    name: str = "adam",
    learning_rate: Union[float, LearningRateSchedule] = 0.001,
    weight_decay: float = 0.0,
    filter_bias_and_bn: bool = True,
    loss_scale: float = 1.0,
    **kwargs: Any,
) -> Optimizer:
    """Create optimizer.

    Args:
        params: Netowrk parameters
        name: Optimizer Name. Default: adam
        learning_rate: Learning rate.
            Accept constant learning rate or a Learning Rate Scheduler. Default: 0.001
        weight_decay: L2 weight decay. Default: 0.
        filter_bias_and_bn: whether to filter batch norm paramters and bias from
            weight decay. If True, weight decay will not apply on BN parameters and
            bias in Conv or Dense layers. Default: True.
        loss_scale: Loss scale in mix-precision training. Default: 1.0
        **kwargs: Arguments feeding to the optimizer

    Returns:
        Optimizer
    """
    if weight_decay and filter_bias_and_bn:
        params = init_group_params(params, weight_decay)

    return entrypoint("optim", name)(
        params=params, learning_rate=learning_rate, loss_scale=loss_scale, **kwargs
    )

from typing import Any, List, Union

from mindspore.nn.learning_rate_schedule import LearningRateSchedule
from mindspore.nn.optim import Adam, Optimizer, SGD


def create_optimizer(
    params: List[Any],
    name: str = "adam",
    learning_rate: Union[float, LearningRateSchedule] = 0.0001,
    loss_scale: float = 1.0,
    weight_decay: float = 0.0001,
    momentum: float = 0.9,
) -> Optimizer:
    """Create optimizer.

    Args:
        params: Netowrk parameters
        name: Optimizer Name. Default: adam
        learning_rate: Learning rate.
            Accept constant learning rate or a Learning Rate Scheduler. Default: 0.0001
        loss_scale: Loss scale in mix-precision training. Default: 1.0
        weight_decay: L2 weight decay. Default: 0.0001
        momentum: momentum in SGD. Default: 0.9

    Returns:
        optimizer: Optimizer
    """
    if name == "adam":
        return Adam(
            params=params,
            learning_rate=learning_rate,
            loss_scale=loss_scale,
            weight_decay=weight_decay,
        )
    elif name == "sgd":
        return SGD(
            params=params,
            learning_rate=learning_rate,
            momentum=momentum,
            loss_scale=loss_scale,
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {name}")

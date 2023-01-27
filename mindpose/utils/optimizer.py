from typing import Any, List

from mindspore.nn.optim import Adam, Optimizer, SGD


def create_optimizer(
    params: List[Any], name: str = "adam", loss_scale: float = 1.0, **kwargs: Any
) -> Optimizer:
    if name == "adam":
        return Adam(params=params, loss_scale=loss_scale, **kwargs)
    elif name == "sgd":
        return SGD(params=params, loss_scale=loss_scale, **kwargs)
    raise ValueError(f"Unsupported optimizer: {name}")

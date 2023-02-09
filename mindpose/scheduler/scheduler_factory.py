from typing import Any, Union

from mindspore.nn.learning_rate_schedule import LearningRateSchedule

from ..register import entrypoint


def create_lr_scheduler(
    name: str,
    lr: float,
    total_epochs: int,
    steps_per_epoch: int,
    warmup: Union[int, float] = 0,
    **kwargs: Any
) -> LearningRateSchedule:
    """Create learning rate scheduler.

    Args:
        name: Name of the scheduler. Default: warmup_cosine_decay
        lr: initial learning rate.
        total_epochs: The number of total epochs of learning rate.
        steps_per_epoch: The number of steps per epoch.
        warmup: If it is a interger, it means the number of warm up steps of
            learning rate. If it is a decimal number, it means the fraction of
            total steps to warm up. Default = 0
        **kwargs: Arguments feed into the corresponding scheduler

    Returns:
        Learning rate scheduler
    """
    return entrypoint("lr_scheduler", name)(
        lr=lr,
        total_epochs=total_epochs,
        steps_per_epoch=steps_per_epoch,
        warmup=warmup,
        **kwargs
    )

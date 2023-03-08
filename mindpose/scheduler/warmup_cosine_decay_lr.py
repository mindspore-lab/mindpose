from typing import Union

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.nn.learning_rate_schedule import LearningRateSchedule

from ..register import register


@register("lr_scheduler", extra_name="warmup_cosine_decay")
class WarmupCosineDecayLR(LearningRateSchedule):
    """CosineDecayLR with warmup.

    Args:
        lr: initial learning rate.
        total_epochs: The number of total epochs of learning rate.
        steps_per_epoch: The number of steps per epoch.
        warmup: If it is a interger, it means the number of warm up steps of
            learning rate. If it is a decimal number, it means the fraction of
            total steps to warm up. Default = 0
        min_lr: Lower lr bound. Default = 0

    Inputs:
        | global_step: Global step

    Outpus:
        | lr: Learning rate at that step
    """

    def __init__(
        self,
        lr: float,
        total_epochs: int,
        steps_per_epoch: int,
        warmup: Union[int, float] = 0,
        min_lr: float = 0.0,
    ) -> None:
        super().__init__()
        total_steps = total_epochs * steps_per_epoch

        if isinstance(warmup, int):
            self.warmup_steps = warmup
        else:
            self.warmup_steps = int(warmup * total_steps)

        if self.warmup_steps > total_steps:
            raise ValueError("Warmup steps must be smaller than total steps")

        self.decay_steps = total_steps - self.warmup_steps
        if self.warmup_steps > 0:
            self.warmup_lr = nn.WarmUpLR(lr, self.warmup_steps)
        self.cosine_decay_lr = nn.CosineDecayLR(min_lr, lr, self.decay_steps)

        self.min_lr = Tensor(min_lr, dtype=ms.float32)

    def step_lr(self, global_step: Tensor) -> Tensor:
        if self.warmup_steps > 0:
            if global_step > self.warmup_steps:
                lr = self.cosine_decay_lr(global_step - self.warmup_steps)
            else:
                lr = self.warmup_lr(global_step)
        else:
            lr = self.cosine_decay_lr(global_step)

        # prevent overflow
        lr = ops.clip_by_value(lr, clip_value_min=self.min_lr)
        return lr

    def construct(self, global_step: Tensor) -> Tensor:
        lr = self.step_lr(global_step)
        return lr

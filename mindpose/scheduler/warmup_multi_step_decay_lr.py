from typing import List, Union

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.nn.learning_rate_schedule import LearningRateSchedule

from ..register import register


@register("lr_scheduler", extra_name="warmup_multi_step_decay")
class WarmupMultiStepDecayLR(LearningRateSchedule):
    """Multi-step decay with warmup.

    Args:
        lr: initial learning rate.
        total_epochs: The number of total epochs of learning rate.
        steps_per_epoch: The number of steps per epoch.
        milestones: The epoch number where the learning rate dacay by one time
        decay_rate: Decay rate. Default = 0.1
        warmup: If it is a interger, it means the number of warm up steps of
            learning rate. If it is a decimal number, it means the fraction of
            total steps to warm up. Default = 0

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
        milestones: List[int],
        decay_rate: float = 0.1,
        warmup: Union[int, float] = 0,
    ) -> None:
        super().__init__()
        total_steps = total_epochs * steps_per_epoch

        if isinstance(warmup, int):
            self.warmup_steps = warmup
        else:
            self.warmup_steps = int(warmup * total_steps)

        if self.warmup_steps > total_steps:
            raise ValueError("Warmup steps must be smaller than total steps")

        if self.warmup_steps > 0:
            self.warmup_lr = nn.WarmUpLR(lr, self.warmup_steps)

        step_lrs = list()
        cur_lr = lr
        k = 0
        for step in range(total_steps):
            if step == milestones[k] * steps_per_epoch:
                cur_lr = cur_lr * decay_rate
                k = min(k + 1, len(milestones) - 1)
            step_lrs.append(cur_lr)
        self.step_lrs = Tensor(step_lrs, dtype=ms.float32)

    def step_lr(self, global_step: Tensor) -> Tensor:
        if self.warmup_steps > 0:
            if global_step > self.warmup_steps:
                lr = self.step_lrs[global_step]
            else:
                lr = self.warmup_lr(global_step)
        else:
            lr = self.step_lrs[global_step]
        return lr

    def construct(self, global_step: Tensor) -> Tensor:
        lr = self.step_lr(global_step)
        return lr

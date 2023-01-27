from typing import Any, List, Union

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.nn.learning_rate_schedule import LearningRateSchedule


class WarmupCosineDecayLR(LearningRateSchedule):
    """CosineDecayLR with warmup
    Args:
        max_lr: (float) upper lr bound for 'WarmupCosineDecayLR' schedulers.
        total_epochs: (int) the number of total epochs of learning rate.
        steps_per_epoch: (int) the number of steps per epoch.
        warmup: If it is a interger, the number of warm up steps of learning rate.
            If it is a decimal number, it is the fraction of total steps to warm up. Default = 0
        min_lr: (float) lower lr bound for 'WarmupCosineDecayLR' schedulers. Default = 0
    """

    def __init__(
        self,
        max_lr: float,
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
            self.warmup_lr = nn.WarmUpLR(max_lr, self.warmup_steps)
        self.cosine_decay_lr = nn.CosineDecayLR(min_lr, max_lr, self.decay_steps)

    def step_lr(self, global_step: Tensor) -> Tensor:
        if self.warmup_steps > 0:
            if global_step > self.warmup_steps:
                lr = self.cosine_decay_lr(global_step - self.warmup_steps)
            else:
                lr = self.warmup_lr(global_step)
        else:
            lr = self.cosine_decay_lr(global_step)
        return lr

    def construct(self, global_step: Tensor) -> Tensor:
        lr = self.step_lr(global_step)
        return lr


class WarmupMultiStepDecayLR(LearningRateSchedule):
    """Multi-step decay with warmup
    Args:
        max_lr: (float) upper lr bound for 'WarmupCosineDecayLR' schedulers.
        total_epochs: (int) the number of total epochs of learning rate.
        steps_per_epoch: (int) the number of steps per epoch.
        milestones (list(int)): The epoch number where the learning rate dacay by one time
        decay_rate (float): decay rate. Default = 0.1
        warmup: If it is a interger, the number of warm up steps of learning rate.
            If it is a decimal number, it is the fraction of total steps to warm up. Default = 0
    """

    def __init__(
        self,
        max_lr: float,
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
            self.warmup_lr = nn.WarmUpLR(max_lr, self.warmup_steps)

        step_lrs = list()
        cur_lr = max_lr
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


def create_lr_scheduler(
    name: str = "warmup_cosine_decay", **kwargs: Any
) -> LearningRateSchedule:
    if name == "warmup_cosine_decay":
        return WarmupCosineDecayLR(**kwargs)
    elif name == "warmup_multi_step_decay":
        return WarmupMultiStepDecayLR(**kwargs)
    else:
        raise ValueError("Unsupported learning rate scheduler: `{name}`")

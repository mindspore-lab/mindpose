import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor


class Allreduce(nn.Cell):
    """Reduces the tensor data across all devices in such a way that all devices will get the same final result."""

    def __init__(self) -> Tensor:
        super(Allreduce, self).__init__()
        self.allreduce_sum = ops.AllReduce(ops.ReduceOp.SUM)

    def construct(self, x: Tensor) -> Tensor:
        return self.allreduce_sum(x)


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = Tensor(0.0, dtype=ms.float32)
        self.avg = Tensor(0.0, dtype=ms.float32)
        self.sum = Tensor(0.0, dtype=ms.float32)
        self.count = Tensor(0.0, dtype=ms.float32)

    def update(self, val: Tensor, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

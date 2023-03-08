import logging
import os
from typing import Any, Dict, Optional

import mindspore as ms
import mindspore.nn as nn
from mindspore import RunContext, SummaryRecord, Tensor
from mindspore.dataset import Dataset
from mindspore.train.callback import Callback

from ..engine.evaluator import Evaluator
from ..engine.inferencer import Inferencer
from ..utils.misc import Allreduce, AverageMeter

_logger = logging.getLogger(__name__)


class EvalCallback(Callback):
    """Running evaluation during training.
    The training, evaluation result will be saved in summary record format for
    visualization. The best and last checkpoint can be saved after each training epoch.

    Args:
        inferencer: Inferencer for running inference on the dataset. Default: None
        evaluator: Evaluator for running evaluation. Default: None
        dataset: The dataset used for running inference. Default: None
        interval: The interval of running evaluation, in epoch. Default: 1
        max_epoch: Total number of epochs for training. Default: 1
        save_best: Saving the best model based on the result of the target metric
            performance. Default: False
        save_last: Saving the last model. Default: False
        best_ckpt_path: Path of the best checkpoint file. Default: "./best.ckpt"
        last_ckpt_path: Path of the last checkpoint file. Default: "./last.ckpt"
        target_metric_name: The metric name deciding the best model to save.
            Default: "AP"
        summary_dir: The directory storing the summary record. Default: "."
        rank_id: Rank id. Default: None
        device_num: Number of devices. Default: None
    """

    def __init__(
        self,
        inferencer: Optional[Inferencer] = None,
        evaluator: Optional[Evaluator] = None,
        dataset: Optional[Dataset] = None,
        interval: int = 1,
        max_epoch: int = 1,
        save_best: bool = False,
        save_last: bool = False,
        best_ckpt_path: str = "./best.ckpt",
        last_ckpt_path: str = "./last.ckpt",
        target_metric_name: str = "AP",
        summary_dir: str = ".",
        rank_id: Optional[int] = None,
        device_num: Optional[int] = None,
    ) -> None:
        self.inferencer = inferencer
        self.evaluator = evaluator

        self.dataset = dataset

        self.interval = interval
        self.max_epoch = max_epoch

        self.save_best = save_best
        self.save_last = save_last

        self.best_ckpt_path = os.path.abspath(best_ckpt_path)
        self.last_ckpt_path = os.path.abspath(last_ckpt_path)
        self.target_metric_name = target_metric_name
        self.summary_dir = summary_dir
        self.rank_id = rank_id if rank_id is not None else 0
        self.device_num = device_num if device_num is not None else 1

        if self.inferencer is None or self.evaluator is None or self.dataset is None:
            _logger.info("Evaluation during training is disabled.")
            self._eval_during_train = False
            if self.save_best:
                _logger.warning(
                    "Best model cannot be saved since `val_while_train` is diabled."
                )
        else:
            self._eval_during_train = True
            if self.target_metric_name not in evaluator.metrics:
                raise ValueError(
                    f"target metric `{self.target_metric_name}` "
                    f"is not listed in evaluator metrics `{evaluator.metrics}`"
                )

        if self.device_num > 1:
            self.all_reduce = Allreduce()

        self.best_result = 0.0
        self.best_epoch = 0.0
        self.summary_record = None

        # store the loss value
        self.loss_meter = AverageMeter()

    def __enter__(self):
        if self.rank_id == 0:
            self.summary_record = SummaryRecord(self.summary_dir)
        return self

    def __exit__(self, *err):
        if self.rank_id == 0:
            self.summary_record.close()

    def on_train_step_end(self, run_context: RunContext) -> None:
        cb_param = run_context.original_args()
        loss = self._get_loss(cb_param)
        loss = loss.asnumpy()
        self.loss_meter.update(loss)

    def on_train_epoch_end(self, run_context: RunContext) -> None:
        cb_param = run_context.original_args()
        cur_epoch = cb_param.cur_epoch_num

        optimizer = cb_param.train_network.network.optimizer
        try:
            lr = optimizer.learning_rate(optimizer.global_step - 1)[0]
        except TypeError:  # constant lr is not callable
            lr = optimizer.learning_rate

        _logger.info(
            f"[rank = {self.rank_id}] epoch = {cur_epoch}, lr = {lr.asnumpy():.3e}, "
            f"loss = {self.loss_meter.avg.asnumpy():.6f}"
        )

        if self.device_num > 1:
            loss_avg = self.all_reduce(self.loss_meter.avg)
            loss_avg /= self.device_num
        else:
            loss_avg = self.loss_meter.avg

        if self.rank_id == 0:
            if self.save_last:
                self._save_last_model(cb_param.train_network)

        output = dict()
        if cur_epoch % self.interval == 0 or cur_epoch == self.max_epoch:
            if self.rank_id == 0 and self._eval_during_train:
                # make sure the inferencer is not for training
                self.inferencer.net.set_train(False)
                result = self.inferencer(self.dataset)
                output = self.evaluator(result)
                _logger.info(output)
                target_result = output[self.target_metric_name]
                if self.save_best:
                    self._save_best_model(
                        cb_param.train_network, target_result, cur_epoch
                    )

        # add summary record
        if self.rank_id == 0:
            for metric_name, metric_value in output.items():
                self.summary_record.add_value(
                    "scalar", "val/" + metric_name, Tensor(metric_value)
                )
            self.summary_record.add_value("scalar", "train/loss", loss_avg)
            self.summary_record.add_value("scalar", "train/lr", lr)
            self.summary_record.add_value("scalar", "epoch", Tensor(cur_epoch))
            self.summary_record.record(cb_param.cur_step_num)
            self.summary_record.flush()

    def _save_best_model(self, net: nn.Cell, result: float, cur_epoch: int) -> None:
        _logger.info(
            f"epoch: {cur_epoch}, "
            f"current result: {result:.3f}, "
            f"previous_best_result: {self.best_result:.3f}."
        )
        if result > self.best_result:
            self.best_result = result
            self.best_epoch = cur_epoch
            ms.save_checkpoint(net, self.best_ckpt_path)
            _logger.info(
                f"Best result is {self.best_result:.3f} at {self.best_epoch} epoch. "
                f"Best checkpoint is saved at `{self.best_ckpt_path}`."
            )
        else:
            _logger.info(
                f"Best result is {self.best_result:.3f} at {self.best_epoch} epoch. "
                "Best checkpoint is unchanged."
            )

    def _save_last_model(self, net: nn.Cell) -> None:
        ms.save_checkpoint(net, self.last_ckpt_path)
        _logger.info(f"Last checkpoint is saved at `{self.last_ckpt_path}`.")

    def _get_loss(self, cb_params: Dict[str, Any]) -> Tensor:
        """
        Get loss from the network output.
        Args:
            cb_params (_InternalCallbackParam): Callback parameters.
        Returns:
            Union[Tensor, None], if parse loss success, will
            return a Tensor value(shape is [1]), else return None.
        """
        output = cb_params.net_outputs
        if output is None:
            _logger.warning(
                "Can not find any output by this network, "
                "so SummaryCollector will not collect loss."
            )
            return None

        if isinstance(output, (int, float, Tensor)):
            loss = output
        elif isinstance(output, (list, tuple)) and output:
            # If the output is a list, since the default network returns loss first,
            # we assume that the first one is loss.
            loss = output[0]
        else:
            _logger.warning(
                "The output type could not be identified, expect type is one of "
                "[int, float, Tensor, list, tuple], "
                "so no loss was recorded in SummaryCollector."
            )
            return None

        if not isinstance(loss, Tensor):
            loss = Tensor(loss)

        loss = loss.mean()
        return loss

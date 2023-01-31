import glob
import json
import logging
import os
import shutil
import time
from typing import Any, Dict, List, Optional

import mindspore as ms
import mindspore.nn as nn
from mindspore import RunContext, SummaryRecord, Tensor
from mindspore.dataset import Dataset
from mindspore.train.callback import Callback

from ..engine.evaluators import Evaluator
from ..engine.inferencer import Inferencer
from ..utils.misc import Allreduce, AverageMeter


class EvalCallback(Callback):
    """Evaluation call back

    Args:
        net_to_save (bool): the network parameter to save when `save_best` is True.
            The default net parameters will be used if it is not provided.
    """

    # TODO: too big, need refactor
    def __init__(
        self,
        inferencer: Inferencer,
        evaluator: Evaluator,
        dataset: Dataset,
        interval: int = 1,
        max_epoch: int = 1,
        net_to_save: Optional[nn.Cell] = None,
        save_best: bool = False,
        save_last: bool = False,
        best_ckpt_path: str = "best.ckpt",
        last_ckpt_path: str = "last.ckpt",
        target_metric_name: str = "AP",
        summary_dir: str = ".",
        rank_id: Optional[int] = None,
        device_num: Optional[int] = None,
        temp_result_dir: str = "/tmp/result",
    ) -> None:
        self.inferencer = inferencer
        self.evaluator = evaluator

        self.dataset = dataset

        self.interval = interval
        self.max_epoch = max_epoch

        if net_to_save is None:
            self.net_to_save = self.net
        else:
            self.net_to_save = net_to_save

        self.save_best = save_best
        self.save_last = save_last
        self.best_ckpt_path = os.path.abspath(best_ckpt_path)
        self.last_ckpt_path = os.path.abspath(last_ckpt_path)
        self.target_metric_name = target_metric_name
        self.summary_dir = summary_dir
        self.rank_id = rank_id if rank_id is not None else 0
        self.device_num = device_num if device_num is not None else 1
        self.temp_result_dir = temp_result_dir

        if self.target_metric_name not in evaluator.get_metrics():
            raise ValueError(
                f"target metric `{self.target_metric_name}` "
                f"is not listed in evaluator metrics `{evaluator.get_metrics()}`"
            )

        if self.device_num > 1:
            self.all_reduce = Allreduce()

        self.best_result = 0.0
        self.best_epoch = 0.0
        self.summary_record = None

        # store the loss value
        self.loss_meter = AverageMeter()

        # clean the result folder
        if os.path.isdir(self.temp_result_dir):
            shutil.rmtree(self.temp_result_dir)

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

        logging.info(
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
                self._save_last_model()

        output = dict()
        if cur_epoch % self.interval == 0 or cur_epoch == self.max_epoch:
            # evaluate on all devices
            result = self.inferencer(self.dataset)
            # accumulate all the result from different rank
            if self.device_num > 1:
                result = self._accumulate_result(cur_epoch, result)

            if self.rank_id == 0:
                output = self.evaluator.evaluate(result)
                target_result = output[self.target_metric_name]
                if self.save_best:
                    self._save_best_model(target_result, cur_epoch)

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

    def _accumulate_result(
        self, epoch: int, result: List[Dict[str, Any]]
    ) -> Optional[List[Dict[str, Any]]]:
        # TODO: use Allgather instead of doing file IO
        outdir = os.path.join(self.temp_result_dir, f"epoch_{epoch}")
        os.makedirs(outdir, exist_ok=True)

        fname = f"rank_{self.rank_id}.json"
        fpath = os.path.join(outdir, fname)

        with open(fpath, "w") as f:
            json.dump(result, f)

        # create a signal once the file is finished writing
        signal_file = f"SUCCESS.{self.rank_id}"
        signal_path = os.path.join(outdir, signal_file)
        with open(signal_path, "w") as f:
            f.write("\n")

        # wait all result finish writing
        pattern = os.path.join(outdir, "SUCCESS.*")
        while len(glob.glob(pattern)) != self.device_num:
            time.sleep(1)

        # accumulate all result to rank 0
        if self.rank_id == 0:
            all_result = []
            pattern = os.path.join(outdir, "rank_*.json")
            files = sorted(glob.glob(pattern))
            for path in files:
                with open(path) as f:
                    all_result.extend(json.load(f))
            return all_result

    def _save_best_model(self, result: float, cur_epoch: int) -> None:
        logging.info(
            f"epoch: {cur_epoch}, "
            f"current result: {result:.3f}, previous_best_result: {self.best_result:.3f}"
        )
        if result > self.best_result:
            self.best_result = result
            self.best_epoch = cur_epoch
            ms.save_checkpoint(self.net_to_save, self.best_ckpt_path)
            logging.info(
                f"Best result is {self.best_result:.3f} at {self.best_epoch} epoch. "
                f"Best checkpoint is saved at {self.best_ckpt_path}"
            )
        else:
            logging.info(
                f"Best result is {self.best_result:.3f} at {self.best_epoch} epoch. "
                "Best checkpoint is unchanged."
            )

    def _save_last_model(self) -> None:
        ms.save_checkpoint(self.net_to_save, self.last_ckpt_path)
        logging.info(f"Last checkpoint is saved at {self.last_ckpt_path}")

    def _get_loss(self, cb_params: Dict[str, Any]) -> Tensor:
        """
        Get loss from the network output.
        Args:
            cb_params (_InternalCallbackParam): Callback parameters.
        Returns:
            Union[Tensor, None], if parse loss success, will return a Tensor value(shape is [1]), else return None.
        """
        output = cb_params.net_outputs
        if output is None:
            logging.warning(
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
            logging.warning(
                "The output type could not be identified, expect type is one of "
                "[int, float, Tensor, list, tuple], so no loss was recorded in SummaryCollector."
            )
            return None

        if not isinstance(loss, Tensor):
            loss = Tensor(loss)

        loss = loss.mean()
        return loss

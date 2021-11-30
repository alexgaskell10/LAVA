import datetime
import logging
import math
import os
import re
import time
import sys
import traceback
from typing import Dict, Optional, List, Tuple, Union, Iterable, Any, Iterator
from allennlp.common.elastic_logger import ElasticLogger
import json
from contextlib import contextmanager
import numpy as np

# try:
#     from apex import amp
# except ImportError:
#     amp = None
import torch
import torch.distributed as dist
import torch.optim.lr_scheduler
from torch.nn.parallel import DistributedDataParallel


from allennlp.common import Lazy, Registrable, Tqdm
from allennlp.common import util as common_util
from allennlp.common.checks import ConfigurationError, check_for_gpu
from allennlp.data import DataLoader
from allennlp.data.dataloader import TensorDict
from allennlp.models.model import Model
from allennlp.nn import util as nn_util
from allennlp.training import util as training_util
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.metric_tracker import MetricTracker
from allennlp.training.momentum_schedulers import MomentumScheduler
from allennlp.training.moving_average import MovingAverage
from allennlp.training.optimizers import Optimizer
from allennlp.training.tensorboard_writer import TensorboardWriter
from allennlp.training.trainer import GradientDescentTrainer, Trainer, BatchCallback, EpochCallback

from .utils import lrange, duplicate_list, description_from_metrics, write_records

import wandb

logger = logging.getLogger(__name__)

@Trainer.register("adversarial_trainer", constructor="from_partial_objects")
class AdversarialTrainer(GradientDescentTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Trains one epoch and returns metrics.
        """
        logger.info("Epoch %d/%d", epoch, self._num_epochs - 1)
        peak_cpu_usage = common_util.peak_memory_mb()
        logger.info(f"Peak CPU memory usage MB: {peak_cpu_usage}")
        gpu_usage = []
        for gpu, memory in common_util.gpu_memory_mb().items():
            gpu_usage.append((gpu, memory))
            logger.info(f"GPU {gpu} memory usage MB: {memory}")

        train_loss = 0.0
        train_reg_loss = 0.0
        # Set the model to "train" mode.
        self._pytorch_model.train()

        # Get tqdm for the training batches
        batch_generator = iter(self.data_loader)
        batch_group_generator = common_util.lazy_groups_of(
            batch_generator, self._num_gradient_accumulation_steps
        )

        logger.info("Training")

        num_training_batches = math.ceil(len(self.data_loader) / self._num_gradient_accumulation_steps)
        batch_group_generator_tqdm = Tqdm.tqdm(batch_group_generator, total=num_training_batches)

        self._last_log = time.time()

        batches_this_epoch = 0
        if self._batch_num_total is None:
            self._batch_num_total = 0

        for batch_group in batch_group_generator_tqdm:
            batches_this_epoch += 1
            self._batch_num_total += 1
            batch_num_total = self._batch_num_total

            self.optimizer.zero_grad()

            batch_group_outputs = []
            for batch in batch_group:
                batch_outputs = self.batch_outputs(batch, for_training=True)
                batch_group_outputs.append(batch_outputs)
                loss = batch_outputs["loss"]
                reg_loss = batch_outputs["reg_loss"]
                if torch.isnan(loss):
                    raise ValueError("nan loss encountered")
                loss = loss / len(batch_group)
                reg_loss = reg_loss / len(batch_group)
                if self._opt_level is not None:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                train_loss += loss.item()
                train_reg_loss += reg_loss.item()

            batch_grad_norm = self.rescale_gradients()
            for parameter in self.model.parameters():
                if parameter.grad is not None:
                    parameter.grad.clamp_(min=-5, max=5)

            # This does nothing if batch_num_total is None or you are using a
            # scheduler which doesn't update per batch.
            if self._learning_rate_scheduler:
                self._learning_rate_scheduler.step_batch(batch_num_total)
            if self._momentum_scheduler:
                self._momentum_scheduler.step_batch(batch_num_total)

            param_updates = None
            self.optimizer.step()

            # Update the description with the latest metrics
            metrics = training_util.get_metrics(
                self.model,
                train_loss,
                train_reg_loss,
                batches_this_epoch,
                world_size=self._world_size,
                cuda_device=[self.cuda_device],
            )

            description = description_from_metrics(metrics)
            batch_group_generator_tqdm.set_description(description, refresh=False)

            self._checkpointer.maybe_save_checkpoint(self, epoch, batches_this_epoch)
            for callback in self._batch_callbacks:
                callback(
                    self,
                    batch_group,
                    batch_group_outputs,
                    epoch,
                    batches_this_epoch,
                    is_training=True,
                )

        metrics = training_util.get_metrics(
            self.model,
            train_loss,
            train_reg_loss,
            batches_this_epoch,
            reset=True,
            world_size=self._world_size,
            cuda_device=[self.cuda_device],
        )
        metrics["cpu_memory_MB"] = peak_cpu_usage
        for (gpu_num, memory) in gpu_usage:
            metrics["gpu_" + str(gpu_num) + "_memory_MB"] = memory

        if 'predictions' in metrics:
            write_records(metrics.pop('predictions'), 'train', epoch, self._serialization_dir)

        return metrics

    def _validation_loss(self, epoch: int) -> Tuple[float, float, int]:
        """
        Computes the validation loss. Returns it and the number of batches.
        """
        logger.info("Validating")

        self._pytorch_model.eval()

        val_generator_tqdm = Tqdm.tqdm(self._validation_data_loader)
        batches_this_epoch = 0
        val_loss = 0
        multiqa_res = {}
        val_reg_loss = 0
        done_early = False
        for batch in val_generator_tqdm:

            batch_outputs = self.batch_outputs(batch, for_training=False)
            loss = batch_outputs.get("loss")
            reg_loss = batch_outputs.get("reg_loss")
            if loss is not None:
                # You shouldn't necessarily have to compute a loss for validation, so we allow for
                # `loss` to be None.  We need to be careful, though - `batches_this_epoch` is
                # currently only used as the divisor for the loss function, so we can safely only
                # count those batches for which we actually have a loss.  If this variable ever
                # gets used for something else, we might need to change things around a bit.
                batches_this_epoch += 1
                val_loss += loss.detach().cpu().numpy()
                if reg_loss is not None:
                    val_reg_loss += reg_loss.detach().cpu().numpy()

            val_metrics = training_util.get_metrics(
                self.model,
                val_loss,
                val_reg_loss,
                batches_this_epoch,
                reset=True,
                world_size=self._world_size,
                cuda_device=[self.cuda_device],
            )

            description = description_from_metrics(val_metrics)
            val_generator_tqdm.set_description(description, refresh=False)

            for callback in self._batch_callbacks:
                callback(
                    self,
                    [batch],
                    [batch_outputs],
                    epoch,
                    batches_this_epoch,
                    is_training=False,
                )

        if 'predictions' in val_metrics:
            write_records(val_metrics.pop('predictions'), 'val', epoch, self._serialization_dir)

        return val_loss, val_reg_loss, batches_this_epoch, val_metrics


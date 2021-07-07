import datetime
import logging
import math
import os
import re
import time
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

from .utils import lrange, duplicate_list

import wandb

logger = logging.getLogger(__name__)

@Trainer.register("custom_trainer", constructor="from_partial_objects")
class CustomTrainer(GradientDescentTrainer):
    def __init__(self, replay_memory, longest_proof, shortest_proof, topk, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._replay_memory = replay_memory
        self._sampler = self.data_loader.batch_sampler.sampler
        self._batch_sampler = self.data_loader.batch_sampler
        self.QLen = None
        self.longest_proof = longest_proof
        self.shortest_proof = shortest_proof
        self.topk = topk

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        for i in [4]:
        # for n,i in enumerate(range(self.shortest_proof, self.longest_proof+1)):
        #     print(f'\n\nBeginning epoch {n} / {self.longest_proof - self.shortest_proof}.\tProof lengths between {self.shortest_proof} - {i} \n\n')
            self.QLen = i + 1
            print('\nBeginning retrieval stage\n')
            retrieval_metrics = self._train_retrieval_epoch(epoch)
            print('\n\n', retrieval_metrics, '\n\n')
            print('\nBeginning binary classification stage\n')
            if i == 2:
                print(2)
            binclass_metrics = self._train_binclass_epoch(epoch)
            print('\n\n', binclass_metrics, '\n\n')
            if i == 1:
                print(1)
            self._replay_memory.empty()
        return retrieval_metrics

    def set_qlen(self):
        self.data_loader.batch_sampler.req_QLens = lrange(1, self.QLen+1) #lrange(1, self.QLen+1) #[self.QLen]
        self._pytorch_model.num_rollout_steps = self.QLen

    def _train_binclass_epoch(self, epoch: int) -> Dict[str, float]:
        """ Trains one epoch and returns metrics.
        """
        mode = 'binary_classification'
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
        self._pytorch_model.set_mode(mode)

        # Configure sampler for binary classification task by passing the ids of the 
        # previous epochs' correctly answered questions to the sampler
        self._sampler.samples = duplicate_list([mem['sampler_idx'] for mem in self._replay_memory], 2)      # TODO: shuffle these lists https://stackoverflow.com/questions/23289547/shuffle-two-list-at-once-with-same-order
        psuedolabels = iter(duplicate_list([mem['topk'] for mem in self._replay_memory], 2))
        if False:
            # # len=1
            # psuedolabels = iter(duplicate_list([[13], [3], [13], [7], [18], [4], [6], [6]], 100))
            # self._sampler.samples = duplicate_list([3, 21, 41, 67, 69, 125, 136, 138], 100)       # Duplicate so a positive and negative binary classification sample can be created from each retrieval sample
            #len=2
            psuedolabels = iter(duplicate_list([[11, 13], [9, 15], [2, 8], [4, 8], [0, 10], [2, 8], [2, 17], [6, 5], [2, 17], [8, 7], [2, 17], [8, 13]], 2))
            self._sampler.samples = duplicate_list([273, 690, 597, 105, 425, 436, 241, 137, 563, 778, 724, 768], 2)

        self._batch_sampler.set_mode(mode)

        # Get tqdm for the training batches
        self.set_qlen()
        batch_generator = iter(self.data_loader)
        batch_group_generator = common_util.lazy_groups_of(
            batch_generator, self._num_gradient_accumulation_steps
        )

        logger.info("Training")

        # num_training_batches = math.ceil(
        #     len(self.data_loader) / self._num_gradient_accumulation_steps
        # )
        num_training_batches = math.ceil(
            len(self._sampler.samples) / self._num_gradient_accumulation_steps
        )
        # Having multiple tqdm bars in case of distributed training will be a mess. Hence only the master's
        # progress is shown
        if self._master:
            batch_group_generator_tqdm = Tqdm.tqdm(
                batch_group_generator, total=num_training_batches
            )
        else:
            batch_group_generator_tqdm = batch_group_generator

        self._last_log = time.time()

        batches_this_epoch = 0
        if self._batch_num_total is None:
            self._batch_num_total = 0

        done_early = False
        for batch_group in batch_group_generator_tqdm:
            if self._distributed:
                # Check whether the other workers have stopped already (due to differing amounts of
                # data in each). If so, we can't proceed because we would hang when we hit the
                # barrier implicit in Model.forward. We use a IntTensor instead a BoolTensor
                # here because NCCL process groups apparently don't support BoolTensor.
                done = torch.tensor(0, device=self.cuda_device)
                torch.distributed.all_reduce(done, torch.distributed.ReduceOp.SUM)
                if done.item() > 0:
                    done_early = True
                    logger.warning(
                        f"Worker {torch.distributed.get_rank()} finishing training early! "
                        "This implies that there is an imbalance in your training "
                        "data across the workers and that some amount of it will be "
                        "ignored. A small amount of this is fine, but a major imbalance "
                        "should be avoided. Note: This warning will appear unless your "
                        "data is perfectly balanced."
                    )
                    break

            batches_this_epoch += 1
            self._batch_num_total += 1
            batch_num_total = self._batch_num_total

            self.optimizer.zero_grad()

            batch_group_outputs = []
            for batch in batch_group:

                for meta in batch['metadata']:
                    meta['psuedolabel_retrievals'] = next(psuedolabels)

                # [batch['metadata'][i]['context'].split('.')[batch['metadata'][i]['psuedolabel_retrievals'][0]].strip() == batch['metadata'][i]['question_text'].strip('.') for i in range(4)]

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
                train_loss = loss.item() * batches_this_epoch
                train_reg_loss += reg_loss.item()

            batch_grad_norm = self.rescale_gradients()

            # This does nothing if batch_num_total is None or you are using a
            # scheduler which doesn't update per batch.
            if self._learning_rate_scheduler:
                self._learning_rate_scheduler.step_batch(batch_num_total)
            if self._momentum_scheduler:
                self._momentum_scheduler.step_batch(batch_num_total)

            param_updates = None
            if self._tensorboard.should_log_histograms_this_batch() and self._master:
                # Get the magnitude of parameter updates for logging.  We need to do some
                # computation before and after the optimizer step, and it's expensive because of
                # GPU/CPU copies (necessary for large models, and for shipping to tensorboard), so
                # we don't do this every batch, only when it's requested.
                param_updates = {
                    name: param.detach().cpu().clone()
                    for name, param in self.model.named_parameters()
                }
                self.optimizer.step()
                for name, param in self.model.named_parameters():
                    param_updates[name].sub_(param.detach().cpu())
            else:
                self.optimizer.step()

            # Update moving averages
            if self._moving_average is not None:
                self._moving_average.apply(batch_num_total)

            # Update the description with the latest metrics
            metrics = training_util.get_metrics(
                self.model,
                train_loss,
                train_reg_loss,
                batches_this_epoch,
                world_size=self._world_size,
                cuda_device=[self.cuda_device],
            )

            # Alon addition - elastic logs - this is a huge Patch ... sorry didn't have time to change this before ACL ..
            # Training logs are saved in training and validation under the training final results
            elastic_metrics = metrics.copy()
            if 'predictions' in elastic_metrics:
                del elastic_metrics['predictions']
            elastic_train_metrics = {'epoch_metrics/'+key:elastic_metrics[key] for key in elastic_metrics}
            elastic_train_metrics.update({'batch_num_total': batch_num_total, 'gpu': self.cuda_device})
            elastic_train_metrics.update({'experiment_name': '/'.join(self._serialization_dir.split('/')[-2:])})
            elastic_train_metrics.pop('optimizer', None)
            elastic_train_metrics.pop('schedule', None)

            if elastic_train_metrics['batch_num_total'] % 100 == 1:
                ElasticLogger().write_log('INFO', 'train_metric', context_dict=elastic_train_metrics)

            # Updating tqdm only for the master as the trainers wouldn't have one
            if self._master:
                description = training_util.description_from_metrics(metrics)
                batch_group_generator_tqdm.set_description(description, refresh=False)

                # TODO
                # self._tensorboard.log_batch(
                #     self.model, self.optimizer, batch_grad_norm, metrics, batch_group, param_updates
                # )

            if self._master:
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

        if self._distributed and not done_early:
            logger.warning(
                f"Worker {torch.distributed.get_rank()} completed its entire epoch (training)."
            )
            # Indicate that we're done so that any workers that have remaining data stop the epoch early.
            done = torch.tensor(1, device=self.cuda_device)
            torch.distributed.all_reduce(done, torch.distributed.ReduceOp.SUM)
            assert done.item()

        # Let all workers finish their epoch before computing
        # the final statistics for the epoch.
        if self._distributed:
            dist.barrier()

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
        return metrics

    def _train_retrieval_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Trains one epoch and returns metrics.
        """
        mode = 'retrieval'
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
        self._pytorch_model.set_mode(mode)

        # Pass the replay buffer to the model
        self._batch_sampler.set_mode(mode)
        self._pytorch_model._replay_memory = self._replay_memory
        self.set_qlen()
        
        # Get tqdm for the training batches
        batch_generator = iter(self.data_loader)
        batch_group_generator = common_util.lazy_groups_of(
            batch_generator, self._num_gradient_accumulation_steps
        )

        logger.info("Training")

        # num_training_batches = math.ceil(
        #     len(self.data_loader) / self._num_gradient_accumulation_steps
        # )
        num_training_batches = math.ceil(
            len(self._sampler.samples) / self._num_gradient_accumulation_steps
        )
        # Having multiple tqdm bars in case of distributed training will be a mess. Hence only the master's
        # progress is shown
        if self._master:
            batch_group_generator_tqdm = Tqdm.tqdm(
                batch_group_generator, total=num_training_batches
            )
        else:
            batch_group_generator_tqdm = batch_group_generator

        self._last_log = time.time()

        batches_this_epoch = 0
        if self._batch_num_total is None:
            self._batch_num_total = 0

        done_early = False
        for batch_group in batch_group_generator_tqdm:
            if self._distributed:
                # Check whether the other workers have stopped already (due to differing amounts of
                # data in each). If so, we can't proceed because we would hang when we hit the
                # barrier implicit in Model.forward. We use a IntTensor instead a BoolTensor
                # here because NCCL process groups apparently don't support BoolTensor.
                done = torch.tensor(0, device=self.cuda_device)
                torch.distributed.all_reduce(done, torch.distributed.ReduceOp.SUM)
                if done.item() > 0:
                    done_early = True
                    logger.warning(
                        f"Worker {torch.distributed.get_rank()} finishing training early! "
                        "This implies that there is an imbalance in your training "
                        "data across the workers and that some amount of it will be "
                        "ignored. A small amount of this is fine, but a major imbalance "
                        "should be avoided. Note: This warning will appear unless your "
                        "data is perfectly balanced."
                    )
                    break

            batches_this_epoch += 1
            self._batch_num_total += 1
            batch_num_total = self._batch_num_total

            self.optimizer.zero_grad()

            batch_group_outputs = []
            for batch in batch_group:

                for m in batch['metadata']:
                    # m['sampler_idx'] = self.data_loader.batch_sampler.batch.pop(0)
                    m['sampler_idx'] = self._batch_sampler.batch.pop(0)

                # assert all([(self.data_loader.dataset.instances[i].fields['metadata'].metadata['id'], batch['metadata'][n]['id']) for n,i in enumerate(self.data_loader.batch_sampler.batch[:4])])

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
                train_loss = loss.item() * batches_this_epoch
                train_reg_loss += reg_loss.item()

            batch_grad_norm = self.rescale_gradients()

            # This does nothing if batch_num_total is None or you are using a
            # scheduler which doesn't update per batch.
            if self._learning_rate_scheduler:
                self._learning_rate_scheduler.step_batch(batch_num_total)
            if self._momentum_scheduler:
                self._momentum_scheduler.step_batch(batch_num_total)

            param_updates = None
            if self._tensorboard.should_log_histograms_this_batch() and self._master:
                # Get the magnitude of parameter updates for logging.  We need to do some
                # computation before and after the optimizer step, and it's expensive because of
                # GPU/CPU copies (necessary for large models, and for shipping to tensorboard), so
                # we don't do this every batch, only when it's requested.
                param_updates = {
                    name: param.detach().cpu().clone()
                    for name, param in self.model.named_parameters()
                }
                self.optimizer.step()
                for name, param in self.model.named_parameters():
                    param_updates[name].sub_(param.detach().cpu())
            else:
                self.optimizer.step()

            # Update moving averages
            if self._moving_average is not None:
                self._moving_average.apply(batch_num_total)

            # Update the description with the latest metrics
            metrics = training_util.get_metrics(
                self.model,
                train_loss,
                train_reg_loss,
                batches_this_epoch,
                world_size=self._world_size,
                cuda_device=[self.cuda_device],
            )

            # Alon addition - elastic logs - this is a huge Patch ... sorry didn't have time to change this before ACL ..
            # Training logs are saved in training and validation under the training final results
            elastic_metrics = metrics.copy()
            if 'predictions' in elastic_metrics:
                del elastic_metrics['predictions']
            elastic_train_metrics = {'epoch_metrics/'+key:elastic_metrics[key] for key in elastic_metrics}
            elastic_train_metrics.update({'batch_num_total': batch_num_total, 'gpu': self.cuda_device})
            elastic_train_metrics.update({'experiment_name': '/'.join(self._serialization_dir.split('/')[-2:])})
            elastic_train_metrics.pop('optimizer', None)
            elastic_train_metrics.pop('schedule', None)

            if elastic_train_metrics['batch_num_total'] % 100 == 1:
                ElasticLogger().write_log('INFO', 'train_metric', context_dict=elastic_train_metrics)

            # Updating tqdm only for the master as the trainers wouldn't have one
            if self._master:
                description = training_util.description_from_metrics(metrics)
                batch_group_generator_tqdm.set_description(description, refresh=False)

                # TODO
                # self._tensorboard.log_batch(
                #     self.model, self.optimizer, batch_grad_norm, metrics, batch_group, param_updates
                # )

            if self._master:
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

        if self._distributed and not done_early:
            logger.warning(
                f"Worker {torch.distributed.get_rank()} completed its entire epoch (training)."
            )
            # Indicate that we're done so that any workers that have remaining data stop the epoch early.
            done = torch.tensor(1, device=self.cuda_device)
            torch.distributed.all_reduce(done, torch.distributed.ReduceOp.SUM)
            assert done.item()

        # Let all workers finish their epoch before computing
        # the final statistics for the epoch.
        if self._distributed:
            dist.barrier()

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
        return metrics

    # def train(self) -> Dict[str, Any]:
    #     try:
    #         epoch_counter = self._restore_checkpoint()
    #     except RuntimeError:
    #         traceback.print_exc()
    #         raise ConfigurationError(
    #             "Could not recover training from the checkpoint.  Did you mean to output to "
    #             "a different serialization directory or delete the existing serialization "
    #             "directory?"
    #         )

    #     training_util.enable_gradient_clipping(self.model, self._grad_clipping)

    #     logger.info("Beginning training.")

    #     val_metrics: Dict[str, float] = {}
    #     this_epoch_val_metric: float = None
    #     metrics: Dict[str, Any] = {}
    #     epochs_trained = 0
    #     training_start_time = time.time()

    #     metrics["best_epoch"] = self._metric_tracker.best_epoch
    #     for key, value in self._metric_tracker.best_epoch_metrics.items():
    #         metrics["best_validation_" + key] = value

    #     best_val_EM = 0
    #     best_val_f1 = 0
    #     for callback in self._epoch_callbacks:
    #         callback(self, metrics={}, epoch=-1)

    #     for epoch in range(epoch_counter, self._num_epochs):
    #         epoch_start_time = time.time()
    #         train_metrics = self._train_epoch(epoch)

    #         # get peak of memory usage
    #         if "cpu_memory_MB" in train_metrics:
    #             metrics["peak_cpu_memory_MB"] = max(
    #                 metrics.get("peak_cpu_memory_MB", 0), train_metrics["cpu_memory_MB"]
    #             )
    #         for key, value in train_metrics.items():
    #             if key.startswith("gpu_"):
    #                 metrics["peak_" + key] = max(metrics.get("peak_" + key, 0), value)

    #         if self._validation_data_loader is not None:
    #             with torch.no_grad():
    #                 # We have a validation set, so compute all the metrics on it.
    #                 val_loss, val_reg_loss, num_batches, val_metrics = self._validation_loss(epoch)

    #                 # It is safe again to wait till the validation is done. This is
    #                 # important to get the metrics right.
    #                 if self._distributed:
    #                     dist.barrier()

    #                 val_metrics = training_util.get_metrics(
    #                     self.model,
    #                     val_loss,
    #                     val_reg_loss,
    #                     num_batches,
    #                     reset=True,
    #                     world_size=self._world_size,
    #                     cuda_device=[self.cuda_device],
    #                 )

    #                 # Check validation metric for early stopping
    #                 this_epoch_val_metric = val_metrics[self._validation_metric]
    #                 self._metric_tracker.add_metric(this_epoch_val_metric)

    #                 if self._metric_tracker.should_stop_early():
    #                     logger.info("Ran out of patience.  Stopping training.")
    #                     break

    #         if self._master and False:
    #             self._tensorboard.log_metrics(
    #                 train_metrics, val_metrics=val_metrics, log_to_console=True, epoch=epoch + 1
    #             )  # +1 because tensorboard doesn't like 0

    #         # Alon addition - elastic logs - this is a huge Patch ... sorry didn't have time to change this before ACL ..
    #         # Training logs are saved in training and validation under the training final results

    #         elastic_val_metrics = val_metrics.copy()
    #         if 'predictions' in elastic_val_metrics:
    #             del elastic_val_metrics['predictions']
    #         elastic_val_metrics = {'validation/' + k: v for k, v in elastic_val_metrics.items()}
    #         elastic_val_metrics.update({'epoch': epoch, 'gpu': self.cuda_device})
    #         elastic_val_metrics.update({'experiment_name': '/'.join(self._serialization_dir.split('/')[-2:])})
    #         elastic_val_metrics.update(
    #             {'optimizer': str(type(self.optimizer)), 'serialization_dir': self._serialization_dir, \
    #              'target_number_of_epochs': self._num_epochs})
    #         elastic_val_metrics.update(self.optimizer.defaults)
    #         elastic_val_metrics.pop('optimizer', None)
    #         elastic_val_metrics.pop('schedule', None)
    #         ElasticLogger().write_log('INFO', 'val_metric', context_dict=elastic_val_metrics)

    #         # Create overall metrics dict
    #         training_elapsed_time = time.time() - training_start_time
    #         metrics["training_duration"] = str(datetime.timedelta(seconds=training_elapsed_time))
    #         metrics["training_start_epoch"] = epoch_counter
    #         metrics["training_epochs"] = epochs_trained
    #         metrics["epoch"] = epoch

    #         for key, value in train_metrics.items():
    #             metrics["training_" + key] = value
    #         for key, value in val_metrics.items():
    #             metrics["validation_" + key] = value

    #         if self._metric_tracker.is_best_so_far():
    #             # Update all the best_ metrics.
    #             # (Otherwise they just stay the same as they were.)
    #             metrics["best_epoch"] = epoch
    #             for key, value in val_metrics.items():
    #                 metrics["best_validation_" + key] = value

    #             self._metric_tracker.best_epoch_metrics = val_metrics

    #         if self._serialization_dir and self._master:
    #             common_util.dump_metrics(
    #                 os.path.join(self._serialization_dir, f"metrics_epoch_{epoch}.json"), metrics
    #             )

    #         # The Scheduler API is agnostic to whether your schedule requires a validation metric -
    #         # if it doesn't, the validation metric passed here is ignored.
    #         if self._learning_rate_scheduler:
    #             self._learning_rate_scheduler.step(this_epoch_val_metric)
    #         if self._momentum_scheduler:
    #             self._momentum_scheduler.step(this_epoch_val_metric)

    #         # TODO
    #         # if self._master and self._save_best_model:
    #         #     self._checkpointer.save_checkpoint(
    #         #         epoch, self, is_best_so_far=self._metric_tracker.is_best_so_far()
    #         #     )

    #         # Wait for the master to finish saving the checkpoint
    #         if self._distributed:
    #             dist.barrier()

    #         for callback in self._epoch_callbacks:
    #             callback(self, metrics=metrics, epoch=epoch)

    #         epoch_elapsed_time = time.time() - epoch_start_time
    #         logger.info("Epoch duration: %s", datetime.timedelta(seconds=epoch_elapsed_time))

    #         if epoch < self._num_epochs - 1:
    #             training_elapsed_time = time.time() - training_start_time
    #             estimated_time_remaining = training_elapsed_time * (
    #                 (self._num_epochs - epoch_counter) / float(epoch - epoch_counter + 1) - 1
    #             )
    #             formatted_time = str(datetime.timedelta(seconds=int(estimated_time_remaining)))
    #             logger.info("Estimated training time remaining: %s", formatted_time)

    #         epochs_trained += 1

    #     # make sure pending events are flushed to disk and files are closed properly
    #     self._tensorboard.close()

    #     # Load the best model state before returning
    #     if self._save_best_model:
    #         best_model_state = self._checkpointer.best_model_state()
    #         if best_model_state:
    #             self.model.load_state_dict(best_model_state)

    #     # ALON last epoch saving results to a file ...
    #     elastic_val_metrics.update(metrics)
    #     if 'predictions' in elastic_val_metrics:
    #         del elastic_val_metrics['predictions']
    #     if 'best_validation_EM' in elastic_val_metrics:
    #         elastic_val_metrics['EM'] = elastic_val_metrics['best_validation_EM']
    #     if 'best_validation_f1' in elastic_val_metrics:
    #         elastic_val_metrics['f1'] = elastic_val_metrics['best_validation_f1']
    #     ElasticLogger().write_log('INFO', 'last_epoch_eval', context_dict=elastic_val_metrics)

    #     # saving a results file:
    #     with open(os.path.join(elastic_val_metrics['serialization_dir'], 'last_epoch_validation_results.json'), 'w') as f:
    #         json.dump(elastic_val_metrics, f)

    #     return metrics

    @classmethod
    def from_partial_objects(
        cls,
        model: Model,
        serialization_dir: str,
        data_loader: DataLoader,
        validation_data_loader: DataLoader = None,
        local_rank: int = 0,
        patience: int = None,
        validation_metric: str = "-loss",
        num_epochs: int = 20,
        cuda_device: int = -1,
        grad_norm: float = None,
        grad_clipping: float = None,
        distributed: bool = None,
        save_best_model: bool = True,
        world_size: int = 1,
        num_gradient_accumulation_steps: int = 1,
        opt_level: Optional[str] = None,
        no_grad: List[str] = None,
        optimizer: Lazy[Optimizer] = None,
        learning_rate_scheduler: Lazy[LearningRateScheduler] = None,
        momentum_scheduler: Lazy[MomentumScheduler] = None,
        tensorboard_writer: Lazy[TensorboardWriter] = None,
        moving_average: Lazy[MovingAverage] = None,
        checkpointer: Lazy[Checkpointer] = None,
        batch_callbacks: List[BatchCallback] = None,
        epoch_callbacks: List[EpochCallback] = None,
        replay_memory = None,
        longest_proof = None, 
        shortest_proof = None, 
        topk = None, 
    ) -> "Trainer":

        """
        This method exists so that we can have a documented method to construct this class using
        `FromParams`. If you are not using `FromParams` or config files, you can safely ignore this
        method.

        The reason we can't just use `__init__` with `FromParams` here is because there are
        sequential dependencies to this class's arguments.  Anything that has a `Lazy[]` type
        annotation needs something from one of the non-`Lazy` arguments.  The `Optimizer` needs to
        have the parameters from the `Model` before it's constructed, and the `Schedulers` need to
        have the `Optimizer`. Because of this, the typical way we construct things `FromParams`
        doesn't work, so we use `Lazy` to allow for constructing the objects sequentially.

        If you're not using `FromParams`, you can just construct these arguments in the right order
        yourself in your code and call the constructor directly.
        """


        check_for_gpu(cuda_device)
        if cuda_device >= 0:
            # Moving model to GPU here so that the optimizer state gets constructed on
            # the right device.
            model = model.cuda(cuda_device)

        if no_grad:
            for name, parameter in model.named_parameters():
                if any(re.search(regex, name) for regex in no_grad):
                    parameter.requires_grad_(False)

        common_util.log_frozen_and_tunable_parameter_names(model)

        parameters = [[n, p] for n, p in model.named_parameters() if p.requires_grad]
        optimizer_ = optimizer.construct(model_parameters=parameters)
        if not optimizer_:
            optimizer_ = Optimizer.default(parameters)

        batches_per_epoch = len(data_loader)  # returns "1" instead of TypeError for _LazyInstances

        moving_average_ = moving_average.construct(parameters=parameters)
        learning_rate_scheduler_ = learning_rate_scheduler.construct(
            optimizer=optimizer_, num_epochs=num_epochs, num_steps_per_epoch=batches_per_epoch
        )
        momentum_scheduler_ = momentum_scheduler.construct(optimizer=optimizer_)

        checkpointer_ = checkpointer.construct() or Checkpointer(serialization_dir)
        tensorboard_writer_ = tensorboard_writer.construct() or TensorboardWriter(serialization_dir)

        return cls(
            replay_memory,
            longest_proof, 
            shortest_proof, 
            topk, 
            model,
            optimizer_,
            data_loader,
            patience=patience,
            validation_metric=validation_metric,
            validation_data_loader=validation_data_loader,
            num_epochs=num_epochs,
            serialization_dir=serialization_dir,
            cuda_device=cuda_device,
            grad_norm=grad_norm,
            grad_clipping=grad_clipping,
            learning_rate_scheduler=learning_rate_scheduler_,
            momentum_scheduler=momentum_scheduler_,
            tensorboard_writer=tensorboard_writer_,
            checkpointer=checkpointer_,
            moving_average=moving_average_,
            batch_callbacks=batch_callbacks,
            epoch_callbacks=epoch_callbacks,
            distributed=distributed,
            local_rank=local_rank,
            world_size=world_size,
            save_best_model = save_best_model,
            num_gradient_accumulation_steps=num_gradient_accumulation_steps,
            opt_level=opt_level,
        )

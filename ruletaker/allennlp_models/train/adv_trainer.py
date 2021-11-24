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

from .utils import lrange, duplicate_list

import wandb

logger = logging.getLogger(__name__)

@Trainer.register("adversarial_trainer", constructor="from_partial_objects")
class AdversarialTrainer(GradientDescentTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _validation_loss(self, epoch):
        self.model.reset('val', epoch)
        val_output = super()._validation_loss(epoch)
        # self.model.reset('val', epoch)
        return val_output

    def _train_epoch(self, epoch):
        self.model.reset('val', epoch)
        if epoch == 1:
            sys.exit()
        train_output = super()._train_epoch(epoch)
        self.model.reset('train', epoch)
        return train_output


import sys
import os
import pandas as pd
from tqdm import tqdm
import yaml
import argparse
import json

import torch
from torch import nn
from torch.utils.data import DistributedSampler, RandomSampler

# from transformers import PreTrainedModel, Trainer, logging
# from transformers.file_utils import is_torch_tpu_available
# from transformers.integrations import is_fairscale_available
# from transformers.models.fsmt.configuration_fsmt import FSMTConfig
# from transformers.optimization import (
#     Adafactor,
#     AdamW,
#     get_constant_schedule,
#     get_constant_schedule_with_warmup,
#     get_cosine_schedule_with_warmup,
#     get_cosine_with_hard_restarts_schedule_with_warmup,
#     get_linear_schedule_with_warmup,
#     get_polynomial_decay_schedule_with_warmup,
# )
# from transformers.trainer_pt_utils import get_tpu_sampler
# from transformers.training_args import ParallelMode

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from transformers.training_args import TrainingArguments

from utils import create_features, collate_fn
from callbacks import CustomFlowCallback

import wandb


class RuleTakerDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, dset, tokenizer):
        self.data_dir = data_dir
        self.dset = dset
        self.exclude_pat = ['birds-electricity', 'NatLang', 'ext']
        self.max_seq_length = 384       # TODO: set properly
        self.tok = tokenizer

        self.data = self.load_features()

    def load_features(self):
        ''' Load json data from files, tokenize and return as 
            features.
        '''
        # Obtain list of files to load
        files = []
        for (dir, _, filenames) in os.walk(self.data_dir):
            if any(pat in dir for pat in self.exclude_pat):
                continue
            for file in filenames:
                if file == self.dset+'.jsonl':
                   files.append(os.path.join(dir, file))

        # Load json data from file
        data = []
        for file in files:
            for line in open(file).readlines():
                row = json.loads(line)
                for q in row['questions']:
                    sample = {'context':row['context'], **q}
                    for k,v in sample.pop('meta').items():
                        sample['meta.'+k] = v
                    data.append(sample)

            break       # TODO: remove to load all data

        # Convert to pd dataframe
        df = pd.DataFrame(data)
        df['cid'] = df.id.apply(lambda id: '-'.join(id.split('-')[:-1]))

        # Convert to features
        df = df.iloc[:100]       # TODO: remove to load all data
        data = create_features(
            df[['label', 'text', 'context']].values, self.max_seq_length, self.tok)

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs):
        ''' Overwrites parent class for custom behaviour during training
        '''
        return super().compute_loss(model, inputs)

    def training_step(self, model, inputs):
        ''' Overwrites parent class for custom behaviour during training
        '''
        return super().training_step(model, inputs)

    def prediction_step(
        self, model, inputs, prediction_loss_only, ignore_keys=None,
    ):
        ''' Overwrites parent class for custom behaviour during prediction
        '''
        return super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys,
        )


def main():
    # Load args from file
    with open('config/test.yaml', 'r') as f:
        manual_args = argparse.Namespace(**yaml.load(f, Loader=yaml.FullLoader))
        args = TrainingArguments(output_dir=manual_args.output_dir)
        for arg in manual_args.__dict__:
            try:
                setattr(args, arg, getattr(manual_args, arg))
            except AttributeError:
                pass
        # with open('config/test.yaml', 'w') as f:
        #     yaml.dump(trainer.args.__dict__, f)

    # model_name = "bert-base-uncased"
    model_name = 'prajjwal1/bert-tiny'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    if args.wandb:
        wandb.init(project="re-re", config=vars(args))

    data = RuleTakerDataset(
        '/vol/bitbucket/aeg19/re-re/data/rule-reasoning-dataset-V2020.2.4', 
        'train', tokenizer)    

    callbacks = [CustomFlowCallback()]

    trainer = CustomTrainer(
        model,
        args=args,
        train_dataset=data,
        eval_dataset=data,
        tokenizer=tokenizer,
        data_collator=collate_fn,
        callbacks=callbacks,
    )

    trainer.train()


if __name__ == '__main__':
    main()
import sys
import os
import pandas as pd
from tqdm import tqdm
import yaml
import argparse
import json
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader

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

os.environ["WANDB_DISABLED"] = "true"

from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, Trainer
from transformers.training_args import TrainingArguments

from utils import (
    create_features, collate_fn, dir_empty_or_nonexistent, compute_metrics
)
from callbacks import CustomFlowCallback


class RuleTakerDataset(torch.utils.data.Dataset):
    def __init__(self, args, dset, tokenizer):
        self.args = args
        self.data_dir = args.data_dir
        self.dset = dset
        self.exclude_pat = ['birds-electricity', 'NatLang', 'ext']
        self.max_seq_length = 512
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
        for file in sorted(files):
            for line in open(file).readlines():
                row = json.loads(line)
                for q in row['questions']:
                    sample = {'context':row['context'], **q}
                    for k,v in sample.pop('meta').items():
                        sample['meta.'+k] = v
                    data.append(sample)

            # if self.args.fast_run:
            #     break
            break

        # Convert to pd dataframe
        df = pd.DataFrame(data)
        df['cid'] = df.id.apply(lambda id: '-'.join(id.split('-')[:-1]))

        # Convert to features
        if self.args.fast_run:
            df = df.iloc[np.random.choice(len(df), 39000)].reset_index(drop=True)
        features = create_features(
            df[['label', 'text', 'context']].values, self.max_seq_length, self.tok)

        return features

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
        # outputs = model(**inputs)
        loss = super().compute_loss(model, inputs)
        return loss

    def training_step(self, model, inputs):
        ''' Overwrites parent class for custom behaviour during training
        '''
        txt = inputs.pop('orig_text')
        # with torch.no_grad():
        #     inputs = {k:v.cuda() for k,v in inputs.items()}
        #     outputs = model(**inputs)
        #     print(compute_metrics(outputs))
        return super().training_step(model, inputs)

    def prediction_step(
        self, model, inputs, prediction_loss_only, ignore_keys=None,
    ):
        ''' Overwrites parent class for custom behaviour during prediction
        '''
        return super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys,
        )

    def log(self, *args):
        ''' Overwrites parent class for custom behaviour during training
        '''
        super().log(*args)

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

        # assert dir_empty_or_nonexistent(args), (
        #     f"Directory exists and not empty:\t{args.output_dir}")

        if args.load_best_model_at_end:
            with open(os.path.join(args.output_dir, 'user_args.yaml'), 'w') as f:
                yaml.dump(manual_args.__dict__, f)
            with open(os.path.join(args.output_dir, 'all_args.yaml'), 'w') as f:
                yaml.dump(args.__dict__, f)

    # model_name = "roberta-large"
    # model_name = "bert-large-uncased"
    model_name = "bert-base-uncased"
    # model_name = "LIAMF-USP/roberta-large-finetuned-race"
    if args.fast_run:
        # model_name = 'prajjwal1/bert-tiny'
        args.load_best_model_at_end = False
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    # model = AutoModel.from_pretrained(model_name)

    if args.wandb:
        import wandb
        wandb.init(project="re-re", config=vars(args))

    # Init dataset
    train = RuleTakerDataset(args, 'train', tokenizer)    
    dev = RuleTakerDataset(args, 'dev', tokenizer)  
    loader_train = DataLoader(train, 
        batch_size=args.per_device_train_batch_size, collate_fn=collate_fn)
    loader_dev = DataLoader(dev, 
        batch_size=args.per_device_eval_batch_size, collate_fn=collate_fn)

    callbacks = [CustomFlowCallback()]

    if args.do_train:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        run_train(0, loader_train, model, torch.device("cuda"), optimizer)
        # trainer = CustomTrainer(
        #     model,
        #     args=args,
        #     train_dataset=train,
        #     eval_dataset=dev,
        #     tokenizer=tokenizer,
        #     data_collator=collate_fn,
        #     callbacks=callbacks,
        #     compute_metrics=compute_metrics,
        # )
        # trainer.train()

    # if args.do_predict:
    #     test = RuleTakerDataset(args, 'test', tokenizer)    
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        run_train(0, loader_train, model, torch.device("cuda"), optimizer)


def run_train(epoch, loader_train, model, device, optimizer):
    model.train()
    model = model.to(device)
    # loader_train = tqdm(loader_train, position=0, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    criterion = nn.CrossEntropyLoss()

    for epoch in range(100):
        cor = 0
        tot = 0
        glob_loss = 0
        for i, input in enumerate(loader_train):
            model.zero_grad()
            model.train()

            label = input.pop('labels').to(device)
            txt = input.pop('orig_text')
            input = {k:v.to(device) for k,v in input.items()}
            out = model(**input)
            # outputs = nn.functional.softmax(out.logits)
            outputs = out.logits
            preds = np.argmax(outputs.cpu().detach(), axis=1)
            labels = torch.zeros_like(out.logits).scatter(1, label.unsqueeze(1), 1)

            loss = criterion(outputs, label.unsqueeze(1).view(-1))
            # loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            # get accuracy
            correct = (outputs.argmax(dim=1) == label).sum().item()
            cor += correct
            tot += outputs.size(0)
            glob_loss += loss
            print(f"acc: {correct/outputs.size(0):.3f}\tculm: {cor/tot:.3f}\tloss={loss:.3f}\tglob_loss={glob_loss*outputs.size(0)/tot:.4f} \tstep: {epoch}.{i/len(loader_train):.2f}")

            lr = optimizer.param_groups[0]['lr']

            # loader_train.set_description(
            #     (f"acc: {correct/outputs.size(0):.3f}\tculm: {cor/tot:.3f}\tloss={loss:.3f}\tglob_loss={glob_loss*outputs.size(0)/tot:.4f} \t{epoch}"))


if __name__ == '__main__':
    main()
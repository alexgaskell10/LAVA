from typing import Dict, Optional, List, Any
import logging
import os
import sys
import time
from math import floor
from copy import deepcopy
from random import shuffle
import random
import numpy as np
from torch._C import Value

try:
    import wandb
except:
    pass

import torch
from torch import nn

from allennlp.models.model import Model

from .utils import (
    safe_log, right_pad, batch_lookup, EPSILON, make_dot, set_dropout, one_hot, lmap, lfilter, print_results, gs, 
    timing
)
from .solver.theory_label_generator import call_theorem_prover_from_lst
from .adversarial import AdversarialGenerator

import logging

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


@Model.register("adversarial_random_benchmark")
class RandomAdversarialBaseline(AdversarialGenerator):
    def __init__(self, benchmark_type=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.benchmark_type = None if benchmark_type in ['none', '-1'] else benchmark_type
        if self.benchmark_type == 'random':
            self.benchmarker = self.random_benchmark
        elif self.benchmark_type == 'word_score':
            self.benchmarker = self.wordscore_benchmark
            self.all_word_overlap_scores = torch.tensor(self.dataset_reader._word_overlap_scores_lst)
            self.mean_overlap_score = self.all_word_overlap_scores.mean()

    # @timing
    def forward(self, phrase=None, label=None, metadata=None, word_overlap_scores=None, **kwargs) -> torch.Tensor:
        if self.training:
            raise ValueError(
                "This model SHOULD NOT BE TRAINED. It is a random benchmark "
                "which does not learn any parameter values."
            )
        self._n_mc = 1
        self._d = phrase['tokens']['token_ids'].device
        qlens = [m['QLen'] for m in metadata]
        qlens_ = [m['QLen'] for m in metadata for _ in range(self._n_mc)]
        lens = torch.tensor(qlens_, device=self._d).unsqueeze(1)
        metadata_orig = [m for m in metadata for _ in range(self._n_mc)]
        orig_sentences = [[m['question_text']] + m['context'].split('.') for m in metadata]
        meta_records = [m['meta_record'] for m in metadata]
        meta_records_orig = [m for m in meta_records for _ in range(self._n_mc)]

        max_facts = max(len(meta['fact_indices']) for meta in metadata)
        max_rules = max(len(meta['rule_indices']) for meta in metadata)
        fact_idxs = torch.tensor([n['fact_indices'] + [-1]*(max_facts-len(n['fact_indices'])) for n in metadata], device=self._d)
        rule_idxs = torch.tensor([n['rule_indices'] + [-1]*(max_rules-len(n['rule_indices'])) for n in metadata], device=self._d)
        
        polarity = torch.tensor([1-int('not' in m['question_text']) for m in metadata], device=self._d)

        nodes = [torch.tensor(m['node_label'][:-1]).nonzero().squeeze(1) for m in metadata]
        max_nodes = max(len(n) for n in nodes)
        nodes = torch.tensor([n.tolist() + [-1]*(max_nodes-len(n)) for n in nodes], device=self._d)
        proof_sentences = [[s[0]]+[s[i+1] for i in n if i!=-1] for n,s in zip(nodes, orig_sentences)]

        # Obtain retrieval logits
        ques_logits, sent_logits, eqiv_logits = self.gen_model(phrase)

        outputs = {"loss": torch.tensor(0.), "estimator": torch.tensor(0.)}

        # qa_output_bl, z_sent_bl, z_eqiv_bl, z_ques_bl, batch_bl, meta_records_bl = self.benchmark(
        #     sent_logits=sent_logits,
        #     ques_logits=ques_logits,
        #     eqiv_logits=eqiv_logits,
        #     metadata=metadata,
        #     label=label,
        #     word_overlap_scores=word_overlap_scores,
        #     meta_records=meta_records,
        #     rule_idxs=rule_idxs,
        # )
 
        z_sent_bl, z_ques_bl, z_eqiv_bl = self.benchmarker(sent_logits, ques_logits, eqiv_logits)

        # Adjust senteqiv samples here
        if self.max_elims > 0:
            z_sent_bl = z_sent_bl.topk(min(self.max_elims, z_sent_bl.size(1)), 1).values
        maskers = [rule_idxs, z_sent_bl] if 'sentence_elimination' in self.adv_perturbations else [rule_idxs, z_sent_bl]
        z_eqiv_bl = self.mask_draws(z_eqiv_bl, *maskers)
        if self.max_flips > 0:
            z_eqiv_bl = z_eqiv_bl.topk(min(self.max_flips, z_eqiv_bl.size(1)), 1).values

        meta_records_bl = meta_records
        metadata_bl = metadata
        if 'sentence_elimination' in self.adv_perturbations:
            meta_records_bl = self.update_meta_sentelim(z_sent_bl, meta_records_bl, metadata_bl)
        if 'question_flip' in self.adv_perturbations:
            meta_records_bl = self.update_meta_quesflip(z_ques_bl, meta_records_bl, metadata_bl)
        if 'equivalence_substitution' in self.adv_perturbations:
            meta_records_bl = self.update_meta_eqivsubt(z_eqiv_bl, meta_records_bl, metadata_bl)

        # engine_labels_bl = call_theorem_prover_from_lst(instances=meta_records_bl)
        # new_labels_bl = [bool(l.item()) if nl is None else nl for nl,l in zip(engine_labels_bl, label)]     # TODO: this is a hack to catch cases where problog cannot solve the logic program. Find a better solution here.
        # modified_label_bl = 1 - torch.tensor(new_labels_bl, device=self._d, dtype=torch.int)

        engine_labels_bl = self.run_tp(meta_records_bl)
        if self.training:
            new_labels_bl = [bool(l.item()) if nl is None else nl for nl,l in zip(engine_labels_bl, label)]     # This is a hack to catch cases where problog cannot solve the logic program. Only happens v. rarely though so not an issue
            skip_ids = []
        else:
            new_labels_bl = [bool(l.item()) if nl is None else nl for nl,l in zip(engine_labels_bl, label)]
            skip_ids = [n for n,e in enumerate(engine_labels_bl) if e is None]
        modified_label_bl = 1 - torch.tensor(new_labels_bl, device=self._d, dtype=torch.int)

        # batch_bl = self._prep_batch(meta_records_bl, label)
        batch_bl = self._prep_batch(meta_records_bl, modified_label_bl)
        with torch.no_grad():
            qa_output_bl = self.qa_model(**batch_bl)
        probs = qa_output_bl["label_probs"]
        preds = probs.argmax(-1)

        new_metadata_bl = batch_bl['metadata']

        if True:
            # Verify meta records sentences are in metadata
            records_sentences_bl = [[i['text'] for i in list(record['triples'].values()) + list(record['rules'].values()) if not i.get('mask',1)] for record in meta_records_bl]
            assert all([all([sent in meta['context'] for sent in sentences]) for meta, sentences in zip(new_metadata_bl, records_sentences_bl)])
            # Verify all metadata sentences are in meta records
            assert all([all([(sent+'.').strip() in sentences for sent in meta['context'].split('.')[:-1]]) for meta, sentences in zip(new_metadata_bl, records_sentences_bl)])

        correct = (preds == modified_label_bl)
        correct[skip_ids] = False
        log_metrics = {}
        log_metrics['n_sentelim'] = (z_sent_bl != -1).sum(-1).float().mean().item()
        log_metrics['n_eqivsubt'] = (z_eqiv_bl != -1).sum(-1).float().mean().item()
        log_metrics['n_quesflip'] = (z_ques_bl != -1).sum(-1).float().mean().item()

        if True:
            sentences = [[m['question_text']] + m['context'].split('.') for m in new_metadata_bl]
            log_metrics['lens'] = np.mean([len(s) for s in sentences])
            records = []
            for i in range(len(correct)):
                i_ = floor(i / self._n_mc)
                record = {
                    "id": metadata_orig[i_]['id'],
                    "qa_fooled": correct[i].item(),
                    "label": label[i_].item(),
                    "mod_label": modified_label_bl[i].item(),     # Target label (i.e. 1-correct label)
                    "polarity": polarity[i_].item(),
                    "question": sentences[i][0],
                    "orig_sentences": orig_sentences[i_],
                    "sampled_sentences": sentences[i],
                    "proof_sentences": proof_sentences[i_][1:],
                    "qa_probs": qa_output_bl["label_probs"][i].cpu().tolist(),
                    "qa_preds": qa_output_bl["label_probs"].argmax(-1)[i].item(),
                    "z_sent": z_sent_bl[i].tolist() if z_sent_bl is not None else None,
                    "z_ques": z_ques_bl[i].tolist() if z_ques_bl is not None else None,
                    "z_eqiv": z_eqiv_bl[i].tolist() if z_eqiv_bl is not None else None,
                    "orig_proof_depth": metadata_orig[i]["QDep"],
                }
                records.append(record)
                record["label"], record["polarity"], record["mod_label"], record["qa_probs"]
            self.records.extend(records)

        self.log_results(qlens_, correct, outputs, log_metrics=log_metrics)

        return outputs
        import pickle as pkl
        batch = pkl.load(open('batch.pkl', 'rb'))
        self.qa_model.to(torch.device('cuda:8'))
        set_dropout(self.qa_model, 0.0)
        with torch.no_grad():
            a = self.qa_model(**batch)
        a['label_logits']

    def _draw_samples_wordscore(self, logits, word_overlap_scores, random_chance=False):
        ''' Obtain samples from a distribution
            - p: probability distribution

            random_chance: sample from uniform distribution
        '''
        mask = (logits == self._p0).all(-1).int()

        tmp = torch.zeros_like(mask, dtype=torch.float)
        tmp[:,:word_overlap_scores.size(1)] = word_overlap_scores
        logits_ = tmp
        scores = torch.clamp(0.5 + logits_ - self.mean_overlap_score, 0, 1)
        samples = torch.bernoulli(scores)
        samples *= 1 - mask       # Ensure padding sentences are not sampled

        # logits_ = logits.view(-1, logits.size(-1))
        # samples = gs(logits_).argmax(-1)
        # mask = (logits_ == -1e9).all(-1).int()
        # samples *= 1 - mask       # Ensure padding sentences are not sampled
        # samples = samples.view(logits.shape[:-1])

        max_draws = samples.sum(-1).max().int()
        z = torch.full_like(samples, -1)[:,:max_draws]
        row, idx = samples.nonzero(as_tuple=True)
        counts = row.bincount()
        for n in range(z.size(0)):
            if n not in row:
                continue
            count = counts[n]
            z[n, :count] = idx[:count]
            idx = idx[count:]

        # Replace padding sentences with -1
        samples = samples * (1-mask.view(samples.shape)) - mask.view(samples.shape)

        return z, samples

    def benchmark(self, 
        sent_logits=None,
        ques_logits=None,
        eqiv_logits=None,
        metadata=None,
        label=None,
        meta_records=None,
        rule_idxs=None,
        **kwargs
    ):
        with torch.no_grad():
            z_sent_bl, z_ques_bl, z_eqiv_bl = self.benchmarker(sent_logits, ques_logits, eqiv_logits)

            # Adjust senteqiv samples here
            if self.max_elims > 0:
                z_sent_bl = z_sent_bl.topk(min(self.max_elims, z_sent_bl.size(1)), 1).values
            maskers = [rule_idxs, z_sent_bl] if 'sentence_elimination' in self.adv_perturbations else [rule_idxs, z_sent_bl]
            z_eqiv_bl = self.mask_draws(z_eqiv_bl, *maskers)
            if self.max_flips > 0:
                z_eqiv_bl = z_eqiv_bl.topk(min(self.max_flips, z_eqiv_bl.size(1)), 1).values

            meta_records_bl = meta_records
            metadata_bl = metadata
            if 'sentence_elimination' in self.adv_perturbations:
                meta_records_bl = self.update_meta_sentelim(z_sent_bl, meta_records_bl, metadata_bl)
            if 'question_flip' in self.adv_perturbations:
                meta_records_bl = self.update_meta_quesflip(z_ques_bl, meta_records_bl, metadata_bl)
            if 'equivalence_substitution' in self.adv_perturbations:
                meta_records_bl = self.update_meta_eqivsubt(z_eqiv_bl, meta_records_bl, metadata_bl)

            batch_bl = self._prep_batch(meta_records_bl, label)
            qa_output_baseline = self.qa_model(**batch_bl)

        return qa_output_baseline, z_sent_bl, z_ques_bl, z_eqiv_bl, batch_bl, meta_records_bl

    def random_benchmark(self, sent_logits, ques_logits, eqiv_logits):
        z_sent_bl, _ = self._draw_samples(sent_logits, random_chance=True)
        z_ques_bl, _ = self._draw_samples(ques_logits, random_chance=True)
        z_eqiv_bl, _ = self._draw_samples(eqiv_logits, random_chance=True)
        return z_sent_bl, z_ques_bl, z_eqiv_bl

    def wordscore_benchmark(self, sent_logits=False, metadata=False, label=False, word_overlap_scores=False, **kwargs):
        raise NotImplementedError
        with torch.no_grad():
            z_baseline, z_bl_onehot = self._draw_samples_wordscore(sent_logits, word_overlap_scores, random_chance=True)
            batch_baseline = self._prep_batch(z_baseline, metadata, label)
            qa_output_baseline = self.qa_model(**batch_baseline)

        return qa_output_baseline, z_baseline, z_bl_onehot, batch_baseline

    @classmethod
    def _load(cls, *args, **kwargs) -> "Model":
        raise NotImplementedError(
            f'{cls} should be freshly initialised as it contains no learned components. '
            '(It has learnable parameters but these are not used - I did not remove these because'
            'it would have been difficult to create this baseline without them.)'
        )

from typing import Dict, Optional, List, Any
import logging
import os
import sys
import time

from transformers.tokenization_auto import AutoTokenizer
import wandb
import random

import torch
from torch.nn.modules.linear import Linear
from torch import batch_norm_gather_stats_with_counts, nn
import torch.nn.functional as F

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy

from transformers import AutoModel

from .retriever_embedders import (
    SpacyRetrievalEmbedder, TransformerRetrievalEmbedder
)
from .utils import safe_log, right_pad, batch_lookup, EPSILON, make_dot, set_dropout, one_hot, lmap, lfilter
from .transformer_binary_qa_model import TransformerBinaryQA
from .baseline import Baseline

torch.manual_seed(0)

@Model.register("variational_inference_base")
class ELBO(Model):
    def __init__(self,
        qa_model: Model,
        variant: str,
        vocab: Vocabulary = None,
        # pretrained_model: str = None,
        requires_grad: bool = True,
        transformer_weights_model: str = None,
        num_labels: int = 2,
        predictions_file=None,
        layer_freeze_regexes: List[str] = None,
        regularizer: Optional[RegularizerApplicator] = None,
        topk: int = 5,
        sentence_embedding_method: str = 'mean',
        dataset_reader = None,
        num_monte_carlo = 1,
    ) -> None:
        super().__init__(qa_model.vocab, regularizer)
        self.variant = variant
        self.qa_model = qa_model        # TODO: replace with fresh transformerbinaryqa
        self.qa_model._loss = nn.CrossEntropyLoss(reduction='none')
        self._loss = nn.CrossEntropyLoss(reduction='none')
        self._mlloss = nn.BCEWithLogitsLoss(reduction='none')
        self.qa_vocab = qa_model.vocab
        self.dataset_reader = dataset_reader
        self.infr_model = InferenceNetwork(variant=variant, vocab=vocab, dataset_reader=dataset_reader)
        self.gen_model = GenerativeNetwork(variant=variant, vocab=vocab, dataset_reader=dataset_reader)
        self.vocab = vocab
        self.regularizer = regularizer
        self.sentence_embedding_method = sentence_embedding_method
        
        self._n_z = topk        # Number of latent retrievals
        self._beta = 0.1        # Scale factor for KL_div TODO: set dynamically
        self._p0 = torch.tensor(-float(1e9))        # TODO: set dynamically
        self._n_mc = num_monte_carlo          # Number of monte carlo steps
        self._logprob_method = 'log-mean-prob'#'log-sum-prob'#'BCE'#'log-sum-prob'      # TODO: set dynamically
        
        self.answers = {}
        # self.x = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self._ = 0
        self._reinforce = Reinforce(baseline_decay=0.99)
        
        set_dropout(self.infr_model, 0.0)
        set_dropout(self.gen_model, 0.0)
        set_dropout(self.qa_model, 0.0)

    def forward(self,
        phrase=None, label=None, metadata=None, retrieval=None, **kwargs,
    ) -> torch.Tensor:
        ''' Forward pass of the network. 

            - p(e|z,q) - answering term (qa_model)
            - q(z|e,q,c) - variational distribution (infr_model)
            - p(z|q,c) - generative distribution (gen_model)
            - e: entailment relation [0,1] (i.e. does q follow from c (or z))
            - z: retrievals (subset of c)
            - c: context (rules + facts)
        '''
        self._d = phrase['tokens']['token_ids'].device
        qlens = [m['QLen'] for m in metadata]
        qlens_ = [m['QLen'] for m in metadata for _ in range(self._n_mc)]
        metadata_ = [m for m in metadata for _ in range(self._n_mc)]
        nodes = torch.cat([torch.tensor(m['node_label']).nonzero() for m in metadata], dim=1).T
        s = self.dataset_reader.decode(phrase['tokens']['token_ids'][0]).split('.')
        flag = False

        infr_logits = self.infr_model(phrase, label)
        gen_logits = self.gen_model(phrase)
        if self._logprob_method == 'BCE':
            # TODO: remove this once bug fixed
            # infr_logits = infr_logits.log_softmax(-1)
            # gen_logits = gen_logits.log_softmax(-1)
            pass
        
        # Take multiple monte carlo samples by tiling the logits
        infr_logits_ = infr_logits.repeat_interleave(self._n_mc, dim=0)
        gen_logits_ = gen_logits.repeat_interleave(self._n_mc, dim=0)
        label_ = label.repeat_interleave(self._n_mc, dim=0)

        z = self._draw_samples(infr_logits, random_chance=False)        # (bsz * mc steps, num retrievals)
        # while (z == 9).any() or (z == 13).any():    # TODO: remove
        #     z = self._draw_samples(infr_logits, random_chance=False)
        if flag:
            if self._ == 0:
            # z = one_hot(z.squeeze(-1), torch.tensor([16]*z.size(0)).view(-1,1).cuda())
                z = torch.tensor([16]*z.size(0)).view(*z.shape).to(self._d)
                self._ = 1
            else:
                z = torch.tensor([15]*z.size(0)).view(*z.shape).to(self._d)
                self._ = 0
            # z = torch.tensor([[9,13]]).to(self._d)
        batch = self._prep_batch(z, metadata_, label_)
        qa_output = self.qa_model(**batch)

        # Compute log probabilites from logits and sample. log probability = -loss
        qa_logprobs = -qa_output['loss']
        infr_logprobs, gen_logprobs = self._compute_logprobs(z, infr_logits_, gen_logits_)

        # Compute REINFORCE estimator for the inference network
        reinforce_reward = qa_logprobs - self._beta * (infr_logprobs - gen_logprobs)
        reinforce_likelihood = self._reinforce(infr_logprobs, reinforce_reward)

        # Compute elbo
        # elbo = qa_logprobs.detach() + self._beta * (gen_logprobs - infr_logprobs) + reinforce_likelihood       # WORKS WITH BUG
        elbo = qa_logprobs.detach() + self._beta * gen_logprobs + reinforce_likelihood       # CORRECT ACCORING TO YISHU
        # elbo = qa_logprobs.detach() + self._beta * gen_logprobs - reinforce_likelihood       # CORRECT ACCORING TO YISHU
        outputs = {"loss": -elbo.mean()}

        with torch.no_grad():
            z_baseline = self._draw_samples(infr_logits, random_chance=True)
            batch_baseline = self._prep_batch(z_baseline, metadata, label)
            qa_output_baseline = self.qa_model(**batch_baseline)
            correct_baseline = (qa_output_baseline["label_probs"].argmax(-1) == label)

        correct = (qa_output["label_probs"].argmax(-1) == label_)
        self.log_results(qlens_, correct, correct_baseline)

        if False:
            self.dataset_reader.decode(phrase['tokens']['token_ids'][0])
            self.dataset_reader.decode(batch['phrase']['tokens']['token_ids'][0]).split('</s> </s>')
            self._pytorch_model.gen_model.model.encoder.layer[0].attention.self.key.weight.grad
            self._pytorch_model.infr_model.model.encoder.layer[0].attention.self.key.weight.grad
            self._pytorch_model.qa_model._transformer_model.encoder.layer[0].attention.self.key.weight.grad
            infr_logits.softmax(-1)[0,nodes[0]]

        if self._n_z == 1 and self._n_mc == 1:
            e = [m['exact_match'] for m in metadata]
            z_ = z.squeeze(-1)
            p = infr_logits.argmax(-1)
            p_ = infr_logits.softmax(-1)
            argmax_ps = p_.gather(1, p.unsqueeze(1)).squeeze()
            action_ps = p_.gather(1, z_.unsqueeze(1)).squeeze()

        # outputs = {"loss": -(self.x - 4)**2}
        return outputs

    def _compute_logprobs(self, z, *logits):
        ''' Compute the log probability of sample z from
            (unnormalized) distribution logit.
        '''
        if self._logprob_method == 'BCE':
            # Using BCE loss (multi-label)
            target = torch.full_like(logits[0], 0).scatter_(1, z, 1.)
            logprobs = [-self._mlloss(logit, target).mean(-1) for logit in logits]
        elif self._logprob_method == 'CE':
            # Using CE loss (single-label): logprob = -CE loss
            logprobs = [-self._loss(logit, z.squeeze(-1)) for logit in logits]
        elif self._logprob_method == 'log-mean-prob':
            # Compute log mean probability. Generalizes CE loss for
            # when retrieving more than one z
            logprobs = []
            for logit in logits:
                # Compute probability for each retrieval
                probs = logit.softmax(-1).gather(1, z)
                # Take mean for avg probability
                mean_probs = probs.mean(-1)
                # Take logs to compute log mean prob
                log_mean_probs = mean_probs.log()
                logprobs.append(log_mean_probs)
        elif self._logprob_method == 'log-sum-prob':
            # Compute log sum probability. Generalizes CE loss for
            # when retrieving more than one z
            logprobs = []
            for logit in logits:
                # Compute probability for each retrieval
                probs = logit.softmax(-1).gather(1, z)
                # Take sum for total probability
                mean_probs = probs.sum(-1)
                # Take logs to compute log sum prob
                log_mean_probs = mean_probs.log()
                logprobs.append(log_mean_probs)
        elif self._logprob_method == 'multivariate-hypergeometric':
            pass
        else:
            raise NotImplementedError
        
        return logprobs

    def log_results(self, qlens, acc, ref=None):
        if ref == None:
            for d, c in zip(qlens, acc):
                if d not in self.answers:
                    self.answers[d] = []
                self.answers[d].append(c.item())

            for d in self.answers.keys():
                all_score = self.answers[d].count(True) / len(self.answers[d])
                last_100 = self.answers[d][-100:].count(True) / len(self.answers[d][-100:])
                print(f'\nL: {d}\tAll: {all_score:.4f}\tLast 100: {last_100:.2f}\tN: {len(self.answers[d])}')
        else:
            for d, c, r in zip(qlens, acc, ref.repeat_interleave(self._n_mc, dim=0)):
                if d not in self.answers:
                    self.answers[d] = [[], []]
                self.answers[d][0].append(c.item())
                self.answers[d][1].append(r.item())

            for d in sorted(self.answers.keys()):
                all_score_a = self.answers[d][0].count(True) / len(self.answers[d][0])
                last_100_a = self.answers[d][0][-100:].count(True) / len(self.answers[d][0][-100:])
                all_score_r = self.answers[d][1].count(True) / len(self.answers[d][1])
                last_100_r = self.answers[d][1][-100:].count(True) / len(self.answers[d][1][-100:])
                print(f'\nM:\tL: {d}\tAll: {all_score_a:.4f}\tLast 400: {last_100_a:.2f}\tB:\tAll: {all_score_r:.4f}\tLast 400: {last_100_r:.2f}\tN: {len(self.answers[d][0])}')
            
    def _prep_batch(self, z, metadata, label):
        ''' Concatenate the latest retrieval to the current 
            query+retrievals. Also update the tensors for the next
            rollout pass.
        '''
        # Concatenate query + retrival to make new query_retrieval matrix of idxs
        sentences = []
        for topk, meta, e in zip(z, metadata, label):
            question = meta['question_text']
            sentence_idxs = topk.tolist()
            context_rtr = [
                toks + '.' for n, toks in enumerate(meta['context'].split('.')[:-1]) 
                if n in sentence_idxs
            ]
            meta['context_str'] = f"q: {question} c: {''.join(context_rtr).strip()}"
            sentences.append((question, ''.join(context_rtr).strip(), e))

        batch = self.dataset_reader.encode_batch(sentences, self.qa_vocab)
        return self.dataset_reader.move(batch, self._d)

    def _gs(self, logits, tau=1):
        ''' Sample using Gumbel Softmax. Ingests raw logits.
        '''
        return F.gumbel_softmax(logits, tau=tau, hard=True, eps=1e-10, dim=-1)

    def _draw_samples(self, p, random_chance=False):
        ''' Obtain samples from a distribution
            - p: probability distribution

            random_chance: sample from uniform distribution
        '''
        # samples = torch.zeros(p.size(0), self._n_z, dtype=torch.long).to(self._d)
        # for i in range(self._n_z):
        #     samples[:, i] = self._gs(p).argmax(-1)
        if random_chance:
            # Overwrite all weights with uniform weight (for masked positions)
            p_ = torch.where(p != self._p0, torch.zeros_like(p), self._p0.to(self._d))
        else:
            # Duplicate tensor for multiple monte carlo samples
            p_ = p.repeat_interleave(self._n_mc, dim=0)        # (bsz * mc steps, num retrievals)
        return torch.multinomial(p_.softmax(-1), self._n_z, replacement=False)
        # return p_.argmax(-1).unsqueeze(1)


class _BaseSentenceClassifier(Model):
    def __init__(self, variant, vocab, dataset_reader, regularizer=None, num_labels=1):
        super().__init__(vocab, regularizer)
        self._predictions = []

        self.variant = variant
        self.dataset_reader = dataset_reader
        self.model = AutoModel.from_pretrained(variant)
        assert 'roberta' in variant     # Only implemented for roberta currently

        transformer_config = self.model.config
        transformer_config.num_labels = num_labels
        self._output_dim = self.model.config.hidden_size

        # unifing all model classification layer
        self.node_class_method = 'mean'     # ['mean', 'concat_first_last']
        self.node_class_k = 1 if self.node_class_method == 'mean' else 2
        # self._W = Linear(self._output_dim * self.node_class_k, num_labels)
        # self._W.weight.data.normal_(mean=0.0, std=0.02)
        # self._W.bias.data.zero_()
        self._W = NodeClassificationHead(self._output_dim * self.node_class_k, 0, num_labels)

        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

        # Split sentences in the context based on full stop
        self.split_idx = self.dataset_reader.encode_token('.', mode='retriever')

        self._p0 = torch.tensor(-float(1e9))

    def forward(self, x) -> torch.Tensor:
        ''' Forward pass of the network. Outputs a distribution over sentences z. 

            - q(z|e,q,c) - variational distribution (infr_model)
            - e: entailment relation [0,1] (i.e. does q follow from c (or z))
            - z: retrievals (subset of c)
            - c: context (rules + facts)
            - q: claim
        '''
        # Compute representations
        embs = self.model(x)[0]

        # Concat first and last token idxs
        max_num_sentences = (x == self.split_idx).nonzero()[:,0].bincount().max() - 1
        node_reprs = torch.full((x.size(0), max_num_sentences), self._p0).to(self._d)      # shape: (bsz, # sentences, 2, model_dim)
        for b in range(x.size(0)):
            # Create list of end idxs of each context item
            end_idxs = (x[b] == self.split_idx).nonzero().squeeze().tolist()
            q_end = end_idxs.pop(0)     # Remove first item as this is the question
            end_idxs.insert(0, q_end + 4)   # Tokenizer adds four "decorative" tokens at the beginning of the context

            # Form tensor containing embedding of first and last token for each sentence
            reprs = torch.zeros(len(end_idxs)-1, self.node_class_k, embs.size(-1)).to(self._d)      # shape: (# context items, 2, model_dim)
            for i in range(len(end_idxs)-1):
                start_idx = end_idxs[i] + 1            # +1 to skip full stop at beginning
                end_idx = end_idxs[i+1] + 1            # +1 to include full stop at end
                
                # Extract reprs for tokens in the sentence from the original encoded sequence
                if self.node_class_method == 'concat_first_last':
                    reprs[i, 0] = embs[b, start_idx]
                    reprs[i, 1] = embs[b, end_idx]
                elif self.node_class_method == 'mean':
                    reprs[i, 0] = embs[b, start_idx:end_idx].mean(dim=0)
                else:
                    raise NotImplementedError

            # Pass through classifier
            reprs_ = reprs.view(reprs.size(0), -1)
            node_logits = self._W(reprs_).squeeze(-1)                
            node_reprs[b, :len(node_logits)] = node_logits

        return node_reprs


class InferenceNetwork(_BaseSentenceClassifier):
    def __init__(self, variant, vocab, dataset_reader, regularizer=None, num_labels=1):
        super().__init__(variant, vocab, dataset_reader, regularizer, num_labels)
        self.e_true = torch.tensor(
            [self.dataset_reader.encode_token(tok, mode='retriever') for tok in '<s> ĠE : ĠTrue </s>'.split()]
        )
        self.e_false = torch.tensor(
            [self.dataset_reader.encode_token(tok, mode='retriever') for tok in '<s> ĠE : ĠFalse </s>'.split()]
        )

    def forward(self, phrase, label, **kwargs) -> torch.Tensor:
        ''' Forward pass of the network. Outputs a distribution over sentences z. 

            - q(z|e,q,c) - variational distribution (infr_model)
            - e: entailment relation [0,1] (i.e. does q follow from c (or z))
            - z: retrievals (subset of c)
            - c: context (rules + facts)
            - q: claim
        '''
        e = label
        qc = phrase['tokens']['token_ids']       # shape = (bsz, context_len)
        self._d = qc.device

        # Prepare model inputs by encoding and concatenating e
        eqc = self._encode_and_append_label(qc, e)

        return super().forward(eqc)

    def _encode_and_append_label(self, encoded, label):
        ''' Tokenizes and appends the label to the already 
            encoded context
        '''
        len_e = len(self.e_true)
        eqc = torch.zeros(encoded.size(0), encoded.size(1) + len_e, dtype=torch.long).to(self._d)
        eqc[:, len_e:] = encoded

        # Add the encoded version of the label "<s> ĠE: [ĠTrue/ĠFalse] </s>"
        for b in range(label.size(0)):
            if label[b] == 1:
                eqc[b, :len_e] = self.e_true.to(self._d)
            elif label[b] == 0:
                eqc[b, :len_e] = self.e_false.to(self._d)
            else:
                raise ValueError
                
        return eqc


class GenerativeNetwork(_BaseSentenceClassifier):
    def __init__(self, variant, vocab, dataset_reader, regularizer=None, num_labels=1):
        super().__init__(variant, vocab, dataset_reader, regularizer, num_labels)

    def forward(self, phrase, **kwargs) -> torch.Tensor:
        ''' Forward pass of the network. Outputs a distribution over sentences z. 

            - q(z|q,c) - generative distribution (gen_model)
            - z: retrievals (subset of c)
            - c: context (rules + facts)
            - q: claim
        '''
        qc = phrase['tokens']['token_ids']       # shape = (bsz, context_len)
        self._d = qc.device

        return super().forward(qc)


class Reinforce(nn.Module):
    """
        A PyTorch module which applies REINFORCE to inputs using a specified reward, and internally
        keeps track of a decaying moving average baseline.

        Parameters
        ----------
        baseline_decay: float, optional (default = 0.99)
            Factor by which the moving average baseline decays on every call.
        
        TODO: from probnmn
    """

    def __init__(self, baseline_decay: float = 0.99):
        super().__init__()
        self._reinforce_baseline = 0.0
        self._baseline_decay = baseline_decay

    def forward(self, inputs, reward):
        # Detach the reward term, we don't want gradients to flow to through it.
        centered_reward = reward.detach() - self._reinforce_baseline
        # centered_reward = reward.detach() - torch.log(torch.tensor(0.5))        # TODO

        # Update moving average baseline.
        self._reinforce_baseline += self._baseline_decay * centered_reward.mean().item()
        return inputs * centered_reward


class NodeClassificationHead(nn.Module):
    """ Head for sentence-level classification tasks.
        TODO: from PRover
    """
    def __init__(self, hidden_size, hidden_dropout_prob, num_labels):
        super(NodeClassificationHead, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

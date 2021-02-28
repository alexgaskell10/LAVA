from math import log
from typing import Dict, Optional, List, Any
import logging
import os
import sys
import time
import wandb

import torch
from torch.nn.modules.linear import Linear
from torch import nn

from allennlp.common.util import sanitize
from allennlp.data import Vocabulary
from allennlp.models.archival import load_archive
from allennlp.models.model import Model
from allennlp.nn import RegularizerApplicator, util
from allennlp.training.metrics import CategoricalAccuracy

from .policy_network import PolicyNetwork
from .utils import safe_log, right_pad, batch_lookup


class RLFramework(Model):
    def __init__(self, qa_model, regularizer, *args, **kwargs):
        super().__init__(qa_model.vocab, regularizer)
        pass

    def prep_episode(self):
        pass


@Model.register("policy_gradients_agent")
class PolicyGradientsAgent(RLFramework):
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
    ) -> None:
        super().__init__(
            qa_model=qa_model,
            variant=variant,
            vocab=vocab,
            # pretrained_model=# pretrained_model,
            requires_grad=requires_grad,
            transformer_weights_model=transformer_weights_model,
            num_labels=num_labels,
            predictions_file=predictions_file,
            layer_freeze_regexes=layer_freeze_regexes,
            regularizer=regularizer,
            topk=topk,
            sentence_embedding_method=sentence_embedding_method,
            dataset_reader=dataset_reader,
        )
        self.pn = PolicyNetwork(
            qa_model,
            variant, 
            sentence_embedding_method, 
            vocab,              # TODO: check right vocab
            dataset_reader,
        )

        self.x = -1
        self.gamma = 1      # TODO
        self.beta = 1       # TODO
        self.n_mc = 5           # TODO
        self.num_rollout_steps = topk
        self.retriever_model = None
        self.run_analysis = False   # TODO
        self.baseline = 'n/a'
        self.training = True

    def reward_fn(self, preds, y):
        return (preds == y).float()

    def forward(self, 
        label: torch.LongTensor = None,
        metadata: List[Dict[str, Any]] = None,
        retrieval: List = None,
        **kwargs,
    ) -> torch.Tensor:
        query = retrieval['tokens']['token_ids'][:,0,:]
        context = retrieval['tokens']['token_ids'][:,1:,:]
        self.device = query.device

        output = self.rollout(query, context, label, metadata)

        def stablize_reward(r):
            r_2D = r.view(-1, self.num_rollouts)
            if self.baseline == 'avg_reward':
                stabled_r_2D = r_2D - r_2D.mean(dim=1, keepdim=True)
            elif self.baseline == 'avg_reward_normalized':
                stabled_r_2D = (r_2D - r_2D.mean(dim=1, keepdim=True)) / (r_2D.std(dim=1, keepdim=True) + ops.EPSILON)
            else:
                raise ValueError('Unrecognized baseline function: {}'.format(self.baseline))
            stabled_r = stabled_r_2D.view(-1)
            return stabled_r
    
        # e1, e2, r = self.format_batch(mini_batch, num_tiles=self.num_rollouts)

        # Compute policy gradient loss
        preds = output['answer_index']
        log_action_probs = output['log_action_probs']
        # action_entropy = output['action_entropy']     # TODO: figure out what this does

        # Compute discounted reward
        final_reward = self.reward_fn(preds, label)
        if self.baseline != 'n/a':
            final_reward = stablize_reward(final_reward)
        cum_discounted_rewards = [0] * self.num_rollout_steps
        cum_discounted_rewards[-1] = final_reward
        R = 0
        for i in range(self.num_rollout_steps - 1, -1, -1):
            R = self.gamma * R + cum_discounted_rewards[i]
            cum_discounted_rewards[i] = R

        # Compute policy gradient
        pg_loss, pt_loss = 0, 0
        for i in range(self.num_rollout_steps):
            log_action_prob = log_action_probs[i]
            pg_loss += -cum_discounted_rewards[i] * log_action_prob
            pt_loss += -cum_discounted_rewards[i] * torch.exp(log_action_prob)

        # # Entropy regularization
        # entropy = torch.cat([x.unsqueeze(1) for x in action_entropy], dim=1).mean(dim=1)
        # pg_loss = (pg_loss - entropy * self.beta).mean()
        # pt_loss = (pt_loss - entropy * self.beta).mean()

        # Entropy regularization
        pg_loss_ = pg_loss.mean()
        pt_loss_ = pt_loss.mean()

        output['model_loss'] = pg_loss_
        output['print_loss'] = float(pt_loss_)
        output['reward'] = final_reward
        # output['entropy'] = float(entropy.mean())
        if self.run_analysis:
            pass
            # fn = torch.zeros(final_reward.size())
            # for i in range(len(final_reward)):
            #     if not final_reward[i]:
            #         if int(preds[i]) in self.kg.all_objects[int(e1[i])][int(r[i])]:
            #             fn[i] = 1
            # output['fn'] = fn

        # Hack to use wandb logging
        if os.environ['WANDB_LOG'] == 'true':
            self.wandb_log(
                output['metadata'], output['label_logits'], output['label'], output['loss']
            )

        return output

    def rollout(self, qr, c, label, metadata):
        ''' Sample traces to obtain expected rewards for an episode.
            - qr: tokenized query + retrieval
            - c: tokenized context
            - label: gold labels
            - metadata: list of dicts
        '''
        # TODO: add an "end" context item of a blank one or something....

        # Storage tensors
        policies = torch.full((self.num_rollout_steps, *c.shape[:-1]), self.x).to(self.device)    # Shape: [num_steps, bsz, max_num_context, max_sentence_len]
        actions = torch.full((self.num_rollout_steps, c.size(0)), self.x).to(self.device)         # Shape: [num_steps, bsz]
        log_action_probs = torch.full_like(actions, self.x)                                       # Shape: [num_steps, bsz]
        
        # Sample for n steps
        for t in range(self.num_rollout_steps - 1):
            policies[t] = self.pn.transit(qr, c)
            actions[t], log_action_probs[t] = self.sample_action(policies[t])
            qr, c, metadata = self.prep_next_batch(qr, c, metadata, actions, t)

        # Answer query
        self.update_meta(qr, metadata, actions)
        output = self.pn.answer(qr, label, metadata)
        policies[-1] = right_pad(output['label_probs'], policies[-1])
        actions[-1], log_action_probs[-1] = self.sample_action(policies[-1])

        # Record trajectory data
        output['policies'] = policies
        output['log_action_probs'] = log_action_probs
        output['samples_actions'] = actions

        return output

    def update_meta(self, query_retrieval, metadata, actions):
        for qr, topk, meta in zip(query_retrieval, actions.T, metadata):
            meta['topk'] = topk.tolist()
            meta['query_retrieval'] = qr.tolist()

    def prep_next_batch(
        self, query_retrieval, context, metadata, actions, t
    ):
        ''' Retrieve the top k context sentences and prepare batch
            for use in the qa model.
        '''
        device = query_retrieval.device
        # Replace retrieved context with padding
        # NOTE torch doesn't permit multidim indexing so first flatten
        # before reshaping
        action_ = (
            actions[t] + torch.arange(actions[t].size(0)).unsqueeze(1).to(device) * context.size(1)
        ).flatten().long()
        context_ = context.clone().contiguous().view(-1, context.size(-1))
        context_[action_] = self.pn.retriever_pad_idx
        context_ = context_.view(context.shape)

        # Concatenate query + retrival to make new query_retrieval
        # tensor of idxs        
        sentences = []
        for topk, meta in zip(actions.T, metadata):
            sentence_idxs = [int(i) for i in topk.tolist() if i != self.x]
            question = meta['question_text']
            context_str = ''.join([
                toks + '.' for n, toks in enumerate(meta['context'].split('.')[:-1]) 
                if n in sentence_idxs
            ]).strip()
            meta['context_str'] = f"q: {question} c: {context_str}"
            sentences.append((question, context_str))
        batch = self.pn.dataset_reader.transformer_indices_from_qa(sentences, self.pn.qa_vocab)
        query_retrieval_ = batch['phrase']['tokens']['token_ids'].to(device)

        return query_retrieval_, context_, metadata

    def sample_action(self, policy):
        action = torch.multinomial(policy, 1)
        action_prob = batch_lookup(policy, action)
        return action.squeeze(), safe_log(action_prob)

    def predict(self):
        pass
    
    def wandb_log(self, metadata, label_logits, label, loss):
        prefix = 'train' if self.training else 'val'

        # Metrics by question depth
        if 'QDep' in metadata[0]:
            depth_accuracies = {}
            retrieval_recalls = {}
            q_depths = torch.tensor([m['QDep'] for m in metadata]).to(label.device)
            for dep in q_depths.unique():
                idxs = (q_depths == dep).nonzero().squeeze()

                # Accuracy
                logits_ = label_logits[idxs]
                labels_ = label[idxs]
                ca = CategoricalAccuracy()
                ca(logits_, labels_)
                depth_accuracies[f"{prefix}_acc_{dep}"] = ca.get_metric()

                # Retrieval recall
                meta = [metadata[i] for i in (idxs if idxs.dim() else idxs.unsqueeze(0)).tolist()]
                retrieval_recalls[f"{prefix}_ret_recall_{dep}"] = self.batch_retrieval_recall(meta)

            wandb.log({**depth_accuracies, **retrieval_recalls}, commit=False)

        # Aggregate metrics
        c = CategoricalAccuracy()
        c(label_logits, label)
        wandb.log({
            prefix+"_loss": loss, 
            prefix+"_acc": self._accuracy.get_metric(), 
            prefix+"_acc_noncuml": c.get_metric(),
            prefix+"_ret_recall": self.batch_retrieval_recall(metadata),
        })


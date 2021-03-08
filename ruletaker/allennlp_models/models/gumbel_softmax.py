from typing import Dict, Optional, List, Any
import logging
import os
import sys
import time
import wandb

import torch
from torch.nn.modules.linear import Linear
from torch import nn
import torch.nn.functional as F

from allennlp.common.util import sanitize
from allennlp.data import Vocabulary
from allennlp.models.archival import load_archive
from allennlp.models.model import Model
from allennlp.nn import RegularizerApplicator, util
from allennlp.training.metrics import CategoricalAccuracy

from .retriever_embedders import (
    SpacyRetrievalEmbedder, TransformerRetrievalEmbedder
)
from .transformer_binary_qa_model import TransformerBinaryQA
from .utils import safe_log, right_pad, batch_lookup, EPSILON, make_dot

from transformers.modeling_roberta import RobertaModel


class CustomRobertaModel(RobertaModel):
    """
    This class overrides :class:`~transformers.BertModel`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """
    def __init__(self, roberta: RobertaModel):
        self.__dict__ = roberta.__dict__

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        index_tensor=None,
    ):
        r"""
    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (:obj:`torch.FloatTensor`: of shape :obj:`(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during pre-training.

            This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import BertModel, BertTokenizer
        import torch

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)

        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

        """

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                causal_mask = causal_mask.to(
                    attention_mask.dtype
                )  # causal and attention masks must have same type with pytorch version < 1.3
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

            if encoder_attention_mask.dim() == 3:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            elif encoder_attention_mask.dim() == 2:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
            else:
                raise ValueError(
                    "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
                        encoder_hidden_shape, encoder_attention_mask.shape
                    )
                )

            encoder_extended_attention_mask = encoder_extended_attention_mask.to(
                dtype=next(self.parameters()).dtype
            )  # fp16 compatibility
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )

        # NOTE: added here
        if index_tensor is not None:
            embedding_output = self.create_grad_pathway(
                tensor=embedding_output, index_tensor=index_tensor
            )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[
            1:
        ]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)

    def create_grad_pathway(embedding_output, index_tensor):
        ''' Creates gradient pathway between the reasoner and
            the retrieval component by bypassing the (non-differentiable)
            indexing during tokenization.
        '''
        i = index_tensor


# class NeuralIndexRobertaEmbeddings(nn.Module):
#     ''' Hack to ensure gradient flow through index operation.
#     '''
#     def __init__(self, embeddings):
#         super().__init__()
#         self.embeddings = embeddings

#     def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
#         return super().forward(input_ids, token_type_ids, position_ids, inputs_embeds)


@Model.register("gumbel_softmax_unified")
class GumbelSoftmaxRetrieverReasoner(Model):
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
        super().__init__(qa_model.vocab, regularizer)
        self.qa_model = qa_model
        self.qa_model._transformer_model = CustomRobertaModel(self.qa_model._transformer_model)
        # self.qa_model._transformer_model.embeddings = NeuralIndexRobertaEmbeddings(self.qa_model._transformer_model.embeddings)
        self.qa_vocab = qa_model.vocab
        self.vocab = vocab
        self.dataset_reader = dataset_reader
        self.variant = variant
        self.sentence_embedding_method = sentence_embedding_method
        self.similarity_func = 'inner'       # TODO: set properly
        self.n_retrievals = 1                # TODO: set properly

        # Rollout params
        self.x = -111
        self.gamma = 1      # TODO
        self.beta = 1       # TODO
        self.n_mc = 5           # TODO
        self.num_rollout_steps = topk
        self.retriever_model = None
        self.run_analysis = False   # TODO
        self.baseline = 'n/a'
        self.training = True
        self._context_embs = None

        self.define_modules()
    
    def define_modules(self):
        if self.variant == 'spacy':
            self.retriever_model = SpacyRetrievalEmbedder(
                sentence_embedding_method=self.sentence_embedding_method,
                vocab=self.vocab,
                variant=self.variant,
            )
            self.tok_name = 'tokens'
            self.retriever_pad_idx = self.vocab.get_token_index(self.vocab._padding_token)      # TODO: standardize these
        elif 'roberta' in self.variant:
            self.retriever_model = TransformerRetrievalEmbedder(
                sentence_embedding_method=self.sentence_embedding_method,
                vocab=self.vocab,
                variant=self.variant,
            )
            self.tok_name = 'token_ids'
            self.retriever_pad_idx = self.dataset_reader.pad_idx(mode='retriever')       # TODO: standardize these
        else:
            raise ValueError(
                f"Invalid retriever_variant: {self.variant}.\nInvestigate!"
            )
        
    def get_retrieval_distr(self, qr, c):
        ''' Compute the probability of retrieving each item given
            the current query+retrieval (i.e. p(zj | zi, y))
        '''
        # Compute embeddings
        e_q = self.get_query_embs(qr)['pooled_output']      # TODO: check against cls_output
        e_c = self.get_context_embs(c)

        # Compute similarities
        if self.similarity_func == 'inner':
            sim = torch.matmul(e_c, e_q.unsqueeze(-1)).squeeze()
        else:
            raise NotImplementedError()

        # Ensure padding receives 0 probability mass
        retrieval_mask = (c != self.retriever_pad_idx).long().unsqueeze(-1)
        similarity = torch.where(
            retrieval_mask.sum(dim=2).squeeze() == 0, 
            torch.tensor(-float("inf")).to(c.device), 
            sim,
        )

        # Policy is distribution over actions
        policy = F.softmax(similarity, dim=1)

        # Deal with nans- these are caused by all sentences being padding.
        # In this case, retrieve following uniform dist
        policy[torch.isnan(policy.sum(dim=1))] = 1 / policy.size(0)
        if torch.isnan(policy.sum(dim=1)).all():
            raise ValueError

        return similarity

    def answer(self, qr, label, metadata, index_tensor):
        return self.get_query_embs(qr, label, metadata, index_tensor)
        
    def get_query_embs(self, qr, label=None, metadata=None, index_tensor=None):
        qr_ = {'tokens': {'token_ids': qr, 'type_ids': torch.zeros_like(qr)}}
        return self.qa_model(qr_, label, metadata, index_tensor)

    @torch.no_grad()
    def get_context_embs(self, c):
        if self._context_embs is None:
            self._context_embs = self.retriever_model(c)
        return self._context_embs
    
    def forward(self, 
        label: torch.LongTensor = None,
        metadata: List[Dict[str, Any]] = None,
        retrieval: List = None,
        **kwargs,
    ) -> torch.Tensor:
        ''' Rollout the forward pass of the network. Consists of
            n retrieval steps followed by an answering step.
        '''
        qr = retrieval['tokens']['token_ids'][:,0,:]
        c = retrieval['tokens']['token_ids'][:,1:,:]
        _d = qr.device

        # TODO: add an "end" context item of a blank one or something....
        # TODO: save the context embeddings from first pass (don't need to recompute)

        # Storage tensors
        unnorm_policies = torch.full((self.num_rollout_steps+1, *c.shape[:-1]), self.x).to(_d)    # Shape: [num_steps, bsz, max_num_context, max_sentence_len]
        actions = torch.full_like(unnorm_policies, self.x)                
        
        # Retrieval rollout phase
        for t in range(self.num_rollout_steps):
            
            # TODO
            unnorm_policies[t] = self.get_retrieval_distr(qr, c)
            # if t == 0:
            #     policy = self.get_retrieval_distr(qr, c)
            #     if not hasattr(self, 'l'):
            #         self.l = nn.Linear(policy.size(1), policy.size(1)).to(policy.device)
            #         self.l.weight.data = torch.ones(self.l.weight.data.shape).to(policy.device) / 1
            #     unnorm_policies[t] = self.l(policy)
            # else:
            #     unnorm_policies[t] = self.get_retrieval_distr(qr, c)

            actions[t] = self.stgs(unnorm_policies[t])
            new_idxs, c, metadata = self.prep_next_batch(c, metadata, actions, t)
            qr = torch.matmul(actions[t].unsqueeze(1), new_idxs).long()     # Indexing is reformulated as matrix-vector product to allow gradient flow. TODO: check this is correct

        # Query answering phase
        self.update_meta(qr, metadata, actions)
        output = self.answer(qr, label, metadata, actions[-1])
        unnorm_policies[-1] = right_pad(output['label_probs'], unnorm_policies[-1])
        actions[t] = self.stgs(unnorm_policies[-1])
        
        # Record trajectory data
        output['unnorm_policies'] = unnorm_policies
        output['sampled_actions'] = actions.argmax(dim=-1)

        self._context_embs = None

        return output

    def stgs(self, logits):
        ''' Straight through gumbel softmax
        '''
        def gs(logits, hard):
            return F.gumbel_softmax(logits, tau=1, hard=hard, eps=1e-10, dim=-1)

        y_soft = gs(logits, False)
        y_hard = gs(logits, True)
        return (y_hard - y_soft).detach() + y_soft

    def update_meta(self, query_retrieval, metadata, actions):
        ''' Log relevant metadata for later use.
        '''
        retrievals = actions.argmax(dim=-1).T
        for qr, topk, meta in zip(query_retrieval, retrievals, metadata):
            meta['topk'] = topk.tolist()
            meta['query_retrieval'] = qr.tolist()

    def prep_next_batch(self, context, metadata, actions, t):
        ''' Concatenate the latest retrieval to the current 
            query+retrievals. Also update the tensors for the next
            rollout pass.
        '''
        # Get indexes of retrieval items
        retrievals = actions.argmax(dim=-1).T

        # Concatenate query + retrival to make new query_retrieval
        # tensor of idxs        
        sentences = []
        for topk, meta in zip(retrievals, metadata):
            question = meta['question_text']
            sentence_idxs = [int(i) for i in topk.tolist()[:t+1] if i != self.x]
            context_str = ''.join([
                toks + '.' for n, toks in enumerate(meta['context'].split('.')[:-1]) 
                if n in sentence_idxs
            ]).strip()
            meta['context_str'] = f"q: {question} c: {context_str}"
            sentences.append((question, context_str))
        batch = self.dataset_reader.transformer_indices_from_qa(sentences, self.qa_vocab)
        src = batch['phrase']['tokens']['token_ids'].unsqueeze(1)

        # Make tensor containing idxs of [query||retrieval]
        _d = context.device
        query_retrieval_ = torch.full((*context.shape[:2], src.size(-1)), self.x, requires_grad=True).to(_d)
        query_retrieval_.scatter_(
            1, retrievals[:,t].repeat(src.size(-1), 1, 1).T, src.float().to(_d)
        )             # [bsz, context items, max sentence len]

        # Replace retrieved context with padding so same context isn't retrieved twice
        current_action = actions[t].argmax(dim=1)
        context.scatter_(
            1, current_action.repeat(context.size(-1), 1, 1).T, self.retriever_pad_idx
        )

        return query_retrieval_, context, metadata

    def predict(self):
        pass

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        if reset == True and not self.training:
            return {
                'EM': self.qa_model._accuracy.get_metric(reset),
                'predictions': self.qa_model._predictions,
            }
        else:
            return {
                'EM': self.qa_model._accuracy.get_metric(reset),
            }

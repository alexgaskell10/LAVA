import torch
from torch import nn

from allennlp.models.model import Model
from allennlp.training.metrics import CategoricalAccuracy
from transformers import AutoModel

from .classification_head import NodeClassificationHead


class _BaseSentenceClassifier(Model):
    def __init__(self, variant, vocab, dataset_reader, has_naf, null_probability, regularizer=None, num_labels=1, dropout=0.):
        super().__init__(vocab, regularizer)
        self._predictions = []
        self._dropout = dropout

        self.variant = variant
        self.dataset_reader = dataset_reader
        self.model = AutoModel.from_pretrained(variant)
        assert 'roberta' in variant     # Only implemented for roberta currently

        transformer_config = self.model.config
        transformer_config.num_labels = num_labels
        self._output_dim = self.model.config.hidden_size

        # unifing all model classification layer
        self.sent_classifier = NodeClassificationHead(self._output_dim, dropout, num_labels)
        self.ques_classifier = NodeClassificationHead(self._output_dim, dropout, num_labels)
        self.eqiv_classifier = NodeClassificationHead(self._output_dim, dropout, num_labels)
        self.naf_layer = nn.Linear(self._output_dim, self._output_dim)
        self.has_naf = has_naf

        self._accuracy = CategoricalAccuracy()
        self._loss = nn.CrossEntropyLoss()
        self.num_labels = num_labels

        # Split sentences in the context based on full stop
        self.split_idx = self.dataset_reader.encode_token('.', mode='retriever')

        self._p0 = null_probability

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
        cls_emb = embs[:,0,:]
        naf_repr = self.naf_layer(cls_emb)

        max_num_sentences = (x == self.split_idx).nonzero()[:,0].bincount().max()
        batch_sent_logits = torch.full((x.size(0), max_num_sentences, self.num_labels), self._p0).to(self._d)       # shape: (bsz, # sentences, 2)
        batch_eqiv_logits = batch_sent_logits.clone()[:,1:,:]                                                               # shape: (bsz, # sentences - 1, 2)   2nd dim is smaller than above because it doesn't need NAF reprs
        batch_ques_logits = batch_sent_logits.clone()[:,:1,:]
        for b in range(x.size(0)):
            # Create list of end idxs of each context item
            end_idxs = (x[b] == self.split_idx).nonzero().squeeze(1).tolist()
            q_end = end_idxs[0]
            q_start = 2             # Beginning of question ix. Ignores BOS tokens
            start_idxs = [q_start, q_end + 4] + end_idxs[1:-1]       # +4 because tok adds four "decorative" tokens at the beginning of the context
            offsets = list(zip(start_idxs, end_idxs))

            # Form tensor containing sentence-level mean of token embeddings
            n_sentences = len(offsets)
            reprs = torch.zeros(n_sentences, 1, embs.size(-1)).to(self._d)      # shape: (# context items, 1, model_dim)
            for i in range(n_sentences):
                start_idx = offsets[i][0] + 1               # +1 to skip full stop at beginning
                end_idx = offsets[i][1] + 1                 # +1 to include full stop at end
                
                # Extract reprs for tokens in the sentence from the original encoded sequence
                reprs[i, 0] = embs[b, start_idx:end_idx].mean(dim=0)

            # Pass through classifier and store results
            ques_reprs = reprs.view(reprs.size(0), -1)[:1]
            sent_reprs = reprs.view(reprs.size(0), -1)[1:]
            sent_reprs = torch.cat((sent_reprs, naf_repr[b].unsqueeze(0)), 0)
            eqiv_reprs = reprs.view(reprs.size(0), -1)[1:]

            ques_logits = self.ques_classifier(ques_reprs).squeeze(-1)
            sent_logits = self.sent_classifier(sent_reprs).squeeze(-1)
            eqiv_logits = self.eqiv_classifier(eqiv_reprs).squeeze(-1)
            
            batch_ques_logits[b, :len(ques_logits)] = ques_logits
            batch_sent_logits[b, :len(sent_logits)] = sent_logits
            batch_eqiv_logits[b, :len(eqiv_logits)] = eqiv_logits

        return batch_ques_logits, batch_sent_logits, batch_eqiv_logits


class GenerativeNetwork(_BaseSentenceClassifier):
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


class GenerativeBaselineNetwork(Model):
    ''' Predicts a single Bernoulli parameter for the whole sample
        (rather than per sentence, as is the case in above GenerativeNetwork)
    '''
    def __init__(self, variant, vocab, dataset_reader, null_probability, dropout=0, regularizer=None, num_labels=1):
        super().__init__(vocab, regularizer)

        self.dataset_reader = dataset_reader
        self.model = AutoModel.from_pretrained(variant)
        assert 'roberta' in variant     # Only implemented for roberta currently

        transformer_config = self.model.config
        transformer_config.num_labels = num_labels
        self._output_dim = self.model.config.hidden_size

        self.out_layer = NodeClassificationHead(self._output_dim, dropout, num_labels)
        self.num_labels = num_labels
        
        self.split_idx = self.dataset_reader.encode_token('.', mode='retriever')
        self._p0 = null_probability

    def forward(self, phrase, **kwargs) -> torch.Tensor:
        
        x = phrase['tokens']['token_ids']
        final_layer_hidden_states = self.model(x)[0]
        out = self.out_layer(final_layer_hidden_states[:,0,:])

        num_sentences = (x == self.split_idx).nonzero()[:,0].bincount()
        node_reprs = torch.cat([out.unsqueeze(1)]*num_sentences.max(), dim=1)
        num_padding = num_sentences.max() - num_sentences

        for n,i in enumerate(num_padding):
            if i == 0:
                continue
            for m in range(1,i+1):
                node_reprs[n,-m] = self._p0

        return node_reprs, None


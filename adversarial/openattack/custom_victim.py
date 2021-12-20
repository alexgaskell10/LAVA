from typing import Callable, Dict, List, Tuple
import torch
import numpy as np

from OpenAttack.victim.classifiers.methods import GetGradient, GetEmbedding
from OpenAttack.tags import Tag, TAG_Classification
from OpenAttack.victim import VictimMethod

from adversarial.openattack.sort_openattack_import import OpenAttack as oa
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


class CustomMethod(VictimMethod):
    def before_call(self, input_):
        pass
    
    def invoke_count(self, input_):
        return len(input_)

class GetPredict(CustomMethod):
    pass

class GetProbability(CustomMethod):
    pass

CLASSIFIER_METHODS : Dict[str, VictimMethod] = {
    "get_pred": GetPredict(),
    "get_prob": GetProbability(),
}


class Classifier(oa.Victim):
    get_pred : Callable[[List[str]], np.ndarray]
    get_prob : Callable[[List[str]], np.ndarray]
    get_grad : Callable[[List[str]], Tuple[np.ndarray, np.ndarray]]
    
    def __init_subclass__(cls):
        invoke_funcs = []
        tags = [ TAG_Classification ]

        for func_name in CLASSIFIER_METHODS.keys():
            if hasattr(cls, func_name):
                invoke_funcs.append((func_name, CLASSIFIER_METHODS[func_name]))
                tags.append( Tag(func_name, "victim") )
                setattr(cls, func_name, CLASSIFIER_METHODS[func_name].method_decorator( getattr(cls, func_name) ) )
        
        super().__init_subclass__(invoke_funcs, tags)


class CustomVictim(Classifier):
    def __init__(self, model, dset_reader, vocab, device):
        self.model = model
        self.dset_reader = dset_reader
        self.vocab = vocab
        self.device = device

    def get_pred(self, input_):
        return self.get_prob(input_).argmax(axis=1)

    # access to the classification probability scores with respect input sentences
    def get_prob(self, input_):
        input = [inp + [1] for inp in input_]
        batch = self.dset_reader.encode_batch(input, self.vocab, True, self.device)
        with torch.no_grad():
            output = self.model(**batch)
        probs = output['label_probs'].detach().cpu().numpy()
        return probs
    
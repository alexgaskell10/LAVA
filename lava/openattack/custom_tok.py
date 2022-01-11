from OpenAttack.text_process.tokenizer.base import Tokenizer
from OpenAttack.data_manager import DataManager
from OpenAttack.tags import *
import nltk

_POS_MAPPING = {
    "JJ": "adj",
    "VB": "verb",
    "NN": "noun",
    "RB": "adv"
}


class CustomTokenizer(Tokenizer):
    """ Tokenizer based on nltk.word_tokenizer.

        :Language: english
    """
    TAGS = { TAG_English }

    def __init__(self, dataset_reader) -> None:
        self.pos_tagger = DataManager.load("TProcess.NLTKPerceptronPosTagger")
        self.dset_reader = dataset_reader
        self.sent_tokenizer = self.dset_reader.tok

    def do_tokenize(self, x, pos_tagging=True):
        tokens = self.sent_tokenizer(x)
        if not pos_tagging:
            return tokens
        ret = []
        for word, pos in self.pos_tagger(tokens):
            if pos[:2] in _POS_MAPPING:
                mapped_pos = _POS_MAPPING[pos[:2]]
            else:
                mapped_pos = "other"
            ret.append( (word, mapped_pos) )
        return ret

    def do_detokenize(self, x):
        return " ".join(x).replace(' .','.')

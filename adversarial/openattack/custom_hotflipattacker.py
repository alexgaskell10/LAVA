import json, os, sys
sys.path.extend(['/vol/bitbucket/aeg19/re-re'])

from typing import List, Optional, Any
from OpenAttack.attackers.hotflip import HotFlipAttacker
from OpenAttack.text_process.tokenizer import Tokenizer, get_default_tokenizer
from OpenAttack.attack_assist.substitute.word import WordSubstitute, get_default_substitute
from OpenAttack.utils import get_language, check_language, language_by_name
from OpenAttack.exceptions import WordNotInDictionaryException
from OpenAttack.tags import Tag
from OpenAttack.attack_assist.filter_words import get_default_filter_words
from OpenAttack.attack_assist.goal import ClassifierGoal
from OpenAttack.tags import *

from adversarial.openattack.custom_victim import Classifier

class CustomHotFlipAttacker(HotFlipAttacker):
    @property
    def TAGS(self):
        return { self.__lang_tag, Tag("get_pred", "victim"), Tag("get_prob", "victim") }
        
    def __init__(self,
        substitute : Optional[WordSubstitute] = None,
        tokenizer : Optional[Tokenizer] = None,
        filter_words : List[str] = [],
        lang = None,
        # vocab = None,
        # device = None,
    ):
        """ HotFlip: White-Box Adversarial Examples for Text Classification. Javid Ebrahimi, Anyi Rao, Daniel Lowd, Dejing Dou. ACL 2018.
            `[pdf] <https://www.aclweb.org/anthology/P18-2006>`__
            `[code] <https://github.com/AnyiRao/WordAdver>`__

            Args:
                tokenizer: A tokenizer that will be used during the attack procedure. Must be an instance of :py:class:`.Tokenizer`
                substitute: A substitute that will be used during the attack procedure. Must be an instance of :py:class:`.WordSubstitute`
                filter_words: A list of words that will be preserved in the attack procesudre.
                lang: The language used in attacker. If is `None` then `attacker` will intelligently select the language based on other parameters.            

            :Classifier Capacity:
                * get_pred
                * get_prob
        """

        lst = []
        if tokenizer is not None:
            lst.append(tokenizer)
        if substitute is not None:
            lst.append(substitute)
        if len(lst) > 0:
            self.__lang_tag = get_language(lst)
        else:
            self.__lang_tag = language_by_name(lang)
            if self.__lang_tag is None:
                raise ValueError("Unknown language `%s`" % lang)
        
        if substitute is None:
            substitute = get_default_substitute(self.__lang_tag)
        self.substitute = substitute

        if tokenizer is None:
            tokenizer = get_default_tokenizer(self.__lang_tag)
        self.tokenizer = tokenizer

        filter_words += get_default_filter_words(self.__lang_tag)
        self.filter_words = set(filter_words)

        check_language([self.tokenizer, self.substitute], self.__lang_tag)

        # self.vocab = vocab
        # self.device = device

    def attack(self, victim: Classifier, input : dict, goal: ClassifierGoal):
        context = input['context']
        question = input['question_text']

        x_orig = self.tokenizer.tokenize(context)
        x_pos =  list(map(lambda x: x[1], x_orig))
        x_orig = list(map(lambda x: x[0], x_orig))

        counter = -1
        for word, pos in zip(x_orig, x_pos):
            counter += 1
            if word in self.filter_words:
                continue
            neighbours = self.get_neighbours(word, pos)
            for neighbour in neighbours:
                x_new = self.tokenizer.detokenize(self.do_replace(x_orig, neighbour, counter))
                pred_target = victim.get_pred([[question, x_new]])[0]
                if goal.check(x_new, pred_target):
                    return x_new, counter
        return None, None
      
    def do_replace(self, x_cur, word, index):
        ret = x_cur[:]
        ret[index] = word
        return ret
                     
    def __call__(self, victim: Classifier, input_: Any):
        if not isinstance(victim, Classifier):
            raise TypeError("`victim` is an instance of `%s`, but `%s` expected" % (victim.__class__.__name__, "Classifier"))
        if Tag("get_pred", "victim") not in victim.TAGS:
            raise AttributeError("`%s` needs victim to support `%s` method" % (self.__class__.__name__, "get_pred"))
        self._victim_check(victim)

        if TAG_Classification not in victim.TAGS:
            raise AttributeError("Victim model `%s` must be a classifier" % victim.__class__.__name__)

        if "target" in input_:
            goal = ClassifierGoal(input_["target"], targeted=True)
        else:
            origin_x = victim.get_pred([ input_["x"] ])[0]
            goal = ClassifierGoal( origin_x, targeted=False )
        
        adversarial_sample, counter = self.attack(victim, input_["x"], goal)

        if adversarial_sample is not None:
            y_adv = victim.get_pred([[input_["x"]["question_text"], adversarial_sample]])[0]
            if not goal.check( adversarial_sample, y_adv ):
                raise RuntimeError("Check attacker result failed: result ([%d] %s) expect (%s%d)" % ( y_adv, adversarial_sample, "" if goal.targeted else "not ", goal.target))
        return adversarial_sample, counter
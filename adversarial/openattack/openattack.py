import json, os, sys
import torch
sys.path.extend(['/vol/bitbucket/aeg19/re-re'])

from adversarial.openattack.sort_openattack_import import OpenAttack as oa
from OpenAttack.attack_assist import filter_words


from ruletaker.allennlp_models.dataset_readers.retrieval_reasoning_reader import RetrievalReasoningReader as DataReader
from adversarial.probe_model import config
from ruletaker.allennlp_models.train.utils import read_pkl
from allennlp.models.archival import load_archive
from allennlp.data.dataloader import DataLoader

from datasets import Dataset

archive = load_archive(**config["archive_config"])
model = archive.model.eval()
vocab = model.vocab
device = torch.device(config["archive_config"]["cuda_device"])

dset = DataReader(**config["dset_config"])
dataset = read_pkl(dset, config["file_path"])
dataset.index_with(vocab)
dataloader = DataLoader(dataset=dataset, **{**config["dataloader_config"], **{"batch_size": 10}})# len(dataset)}})      # TODO

dset_dict = list(dataloader)[0]

hf_dataset = Dataset.from_dict({
    'label': dset_dict['label'].data,
    'metadata': dset_dict['metadata'],
    'token_ids': dset_dict['phrase']['tokens']['token_ids'],
    'mask': dset_dict['phrase']['tokens']['mask'],
    'type_ids': dset_dict['phrase']['tokens']['type_ids'],
})



from adversarial.openattack.custom_victim import CustomVictim
victim = CustomVictim(model, dset, vocab, device)

def dataset_mapping(x):
    return {
        "x": {"question_text": x["metadata"]["question_text"], "context": x["metadata"]["context"]},
        "target": 1 - x["label"],
        "y": x["label"],
    }

hf_dataset = hf_dataset.map(function=dataset_mapping)

from adversarial.openattack.custom_tok import CustomTokenizer
tok = CustomTokenizer(dset)

from adversarial.openattack.custom_hotflipattacker import CustomHotFlipAttacker
from adversarial.openattack.custom_textfoolerattacker import CustomTextFoolerAttacker
filter_words = ['Anne', 'Bob', 'Charlie', 'Dave', 'Erin', 'Fiona', 'Gary', 'Harry']
# attacker = CustomHotFlipAttacker(tokenizer=tok, filter_words=filter_words)
attacker = CustomTextFoolerAttacker(tokenizer=tok, filter_words=filter_words)

from adversarial.openattack.custom_attackeval import CustomAttackEval
attack_eval = CustomAttackEval(attacker, victim, tokenizer=tok)
outputs = attack_eval.eval(hf_dataset, visualize=True)
results = outputs[1]

import pickle as pkl
with open('test_save.pkl', 'wb') as f:
    pkl.dump(results, f)

print('Done')








# import OpenAttack as oa
# import numpy as np
# import datasets
# import nltk
# from nltk.sentiment.vader import SentimentIntensityAnalyzer

# # configure access interface of the customized victim model by extending OpenAttack.Classifier.
# class MyClassifier(oa.Classifier):
#     def __init__(self):
#         # nltk.sentiment.vader.SentimentIntensityAnalyzer is a traditional sentiment classification model.
#         nltk.download('vader_lexicon')
#         self.model = SentimentIntensityAnalyzer()
    
#     def get_pred(self, input_):
#         return self.get_prob(input_).argmax(axis=1)

#     # access to the classification probability scores with respect input sentences
#     def get_prob(self, input_):
#         ret = []
#         for sent in input_:
#             # SentimentIntensityAnalyzer calculates scores of “neg” and “pos” for each instance
#             res = self.model.polarity_scores(sent)

#             # we use 𝑠𝑜𝑐𝑟𝑒_𝑝𝑜𝑠 / (𝑠𝑐𝑜𝑟𝑒_𝑛𝑒𝑔 + 𝑠𝑐𝑜𝑟𝑒_𝑝𝑜𝑠) to represent the probability of positive sentiment
#             # Adding 10^−6 is a trick to avoid dividing by zero.
#             prob = (res["pos"] + 1e-6) / (res["neg"] + res["pos"] + 2e-6)

#             ret.append(np.array([1 - prob, prob]))
        
#         # The get_prob method finally returns a np.ndarray of shape (len(input_), 2). See Classifier for detail.
#         return np.array(ret)

# def dataset_mapping(x):
#     return {
#         "x": x["sentence"],
#         "y": 1 if x["label"] > 0.5 else 0,
#     }
    
# # load some examples of SST-2 for evaluation
# dataset = datasets.load_dataset("sst", split="train[:20]").map(function=dataset_mapping)
# # choose the costomized classifier as the victim model
# victim = MyClassifier()
# # choose PWWS as the attacker and initialize it with default parameters
# # attacker = oa.attackers.PWWSAttacker()
# attacker = oa.attackers.TextFoolerAttacker()
# # prepare for attacking
# attack_eval = oa.AttackEval(attacker, victim)
# # launch attacks and print attack results 
# attack_eval.eval(dataset, visualize=True)
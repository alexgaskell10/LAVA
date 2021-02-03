from allennlp.data.tokenizers import SpacyTokenizer
from allennlp.common.util import get_spacy_model

spacy = get_spacy_model(spacy_model_name="en_core_web_md", pos_tags=False, parse=False, ner=False)
x = "I like to walk outside and Bob eats food and Vanessa does not."
y = spacy(x)
print(y)
pass

# y[1].cluster = 1684
# like = y[1].cluster
vector = spacy.vocab.vectors
for s in y:
    v = vector.data[vector.key2row[s.lemma]]
    z = s.vector
    assert not list(v == z).count(False)

pass
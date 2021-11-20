import torch
from torch import nn
from torch.nn import functional as F

logits = torch.tensor([[-0.1,0.2,0.3]], dtype=torch.float)       # tensor([0.0900, 0.2447, 0.6652])
n = int(1e5)
F.gumbel_softmax(logits.repeat(n, 1), tau=0.1, hard=True, eps=1e-10, dim=-1).argmax(-1).bincount().float() / n
F.gumbel_softmax(logits.repeat(n, 1), tau=100, hard=True, eps=1e-10, dim=-1).argmax(-1).bincount().float() / n
# torch.multinomial(logits.softmax(-1), 10000, replacement=True)

# tau = 0.1
# gumbels = -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()  # ~Gumbel(0,1)
# gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
# y_soft = gumbels.softmax(dim)


from rouge_score.rouge_scorer import RougeScorer

scorer = RougeScorer(["rouge1"])
print(scorer.score("the cat sat on the mat", "the cold dark dog was brown")["rouge1"].fmeasure)

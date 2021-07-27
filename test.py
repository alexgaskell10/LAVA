import torch
from torch import nn
from torch.nn import functional as F

logits = torch.tensor([1,2,3], dtype=torch.float)       # tensor([0.0900, 0.2447, 0.6652])
F.gumbel_softmax(logits.view(1,-1).repeat(100, 1), tau=1, hard=True, eps=1e-10, dim=-1).argmax(-1)
# torch.multinomial(logits.softmax(-1), 10000, replacement=True)
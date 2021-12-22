import torch
from torch import nn


class NodeClassificationHead(nn.Module):
    def __init__(self, dim, dropout, n_labels):
        super().__init__()
        self.dense = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(dim, n_labels)

    def forward(self, input):
        x = self.dropout(input)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


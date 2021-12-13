import torch
from torch import nn


class NodeClassificationHead(nn.Module):
    def __init__(self, dim, dropout, n_labels):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.out_fc = nn.Linear(dim, n_labels)

    def forward(self, input):
        x = self.dropout(input)
        x = self.fc(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_fc(x)
        return x


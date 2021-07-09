import torch
from torch import nn
import torch.nn.functional as F

class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.b = nn.Parameter(torch.tensor(0.12), requires_grad=True)
        # self.sb = SampleBaseline()

    def forward(self, *args, **kwargs):
        return self.b


# class SampleBaseline(nn.Module):
#     def __init__(self):
#         super().__init__()
        

#     def forward(self, *args, **kwargs):
#         pass

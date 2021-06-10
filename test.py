import torch
from torch import nn
from torch.nn import functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(20, 10)
        self.l2 = nn.Linear(20, 10)

    def forward(self, x, z):
        e_c = self.l1(x)
        e_z = self.embed(z)
        sim = torch.matmul(e_z, e_c.T).squeeze(-1)
        return sim

    @torch.no_grad()
    def embed(self, z):
        return self.l2(z)


x = torch.rand((1, 20), requires_grad=True)
z = torch.rand((1, 8, 20), requires_grad=False)
y = torch.tensor([4])

model = Model()
learning_rate = 1e-2
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

for _ in range(100):
    pred = model(x, z)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

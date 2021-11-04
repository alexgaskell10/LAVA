import torch
from torch import nn

x = torch.randn(1,5, requires_grad=True)
target = torch.tensor([0.,0.,0.])

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.W2 = nn.Linear(5,1)
        self.W1 = nn.Linear(5,5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.W1(x)
        x = self.relu(x)
        x = self.W2(x)
        return x

bce = nn.BCEWithLogitsLoss(reduction='none')
ce = nn.CrossEntropyLoss(reduction='none')
model1 = Model()
model2 = Model()
optimizer = torch.optim.Adam(model1.parameters(), lr=0.01)
# loss = nn.KLDivLoss(reduction='none')

for _ in range(1000):
    # optimizer.zero_grad()
    logits1 = model1(x)
    logits2 = 5 #model2(x)
    # l = loss(logits1, logits2).mean()
    l = (logits1 - logits2)**2
    l.backward()
    print(logits1.item(), l.item())

    # bce_loss = bce(logits.squeeze(), target.squeeze())          #logits.sigmoid().log() * target + (1-target)*(1-logits.sigmoid()).log()
    # ce_loss = ce(logits, target.argmax(-1).unsqueeze(0))        #-logits.log_softmax(-1)[0, target.argmax(-1)]
    # loss = bce_loss.mean()
    # loss.backward()
    optimizer.step(),

print('Done')
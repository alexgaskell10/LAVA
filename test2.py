import torch
from torch import nn
from torch.nn import functional as F

from allennlp.models.archival import load_archive
from ruletaker.allennlp_models import TransformerBinaryQA
from transformers import AutoModel, AutoTokenizer

def data():
    data = {"id":"RelNeg-D5-168","context":"The lion likes the bald eagle. If someone eats the lion then the lion eats the bald eagle. If someone is young then they like the dog. The bald eagle likes the rabbit. If someone is young and they visit the lion then the lion visits the dog. If someone eats the dog and they visit the dog then they eat the lion. If someone is round and cold then they eat the dog. The rabbit likes the bald eagle. The lion does not eat the rabbit. The dog does not like the bald eagle. The lion is cold. The lion visits the rabbit. The rabbit is young. The rabbit visits the dog. The bald eagle is young. The dog is round. The rabbit eats the bald eagle. If someone eats the bald eagle then they visit the lion. The lion is round. The bald eagle likes the lion.","meta":{"sentenceScramble":[9,16,19,3,17,20,15,13,6,5,7,10,12,14,1,4,11,18,8,2]},"questions":[{"id":"RelNeg-D5-168-1","text":"The rabbit eats the bald eagle.","label":True,"meta":{"QDep":0,"QLen":1,"strategy":"proof","Qid":"Q1"}}]}
    return data["context"], data['questions'][0]['text'], 1

class Model(nn.Module):
    def __init__(self, qa_model):
        super().__init__()
        self.qa_model = qa_model
        self.retriever = AutoModel.from_pretrained(qa_model._pretrained_model)
        self.tokenizer = AutoTokenizer.from_pretrained(qa_model._pretrained_model)

    def forward(self, context, question):
        context = [c + '.' for c in context.split('.')[:-1]]
        x = self.tokenizer.batch_encode_plus([question], return_tensors='pt')
        z = self.tokenizer.batch_encode_plus(context, return_tensors='pt', pad_to_max_length=True)
        x = {'tokens':{'token_ids': x['input_ids'], 'type_ids':torch.zeros_like(x['input_ids'])}}

        e_q = self.qa_model(x)['pooled_output']
        e_c = self.embed(z)[0][:,0,:].squeeze().unsqueeze(0)

        sim = torch.matmul(e_c, e_q.T).squeeze(-1)
        return sim

    @torch.no_grad()
    def embed(self, z):
        return self.retriever(z['input_ids'])


def train(model):
    context, question, _ = data()
    y = torch.tensor([16])

    learning_rate = 1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(100):
        pred = model(context, question)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss)


ruletaker_archive = "ruletaker/runs/depth-5-base/model.tar.gz"
archive = load_archive(ruletaker_archive, -1)
model = Model(archive.model)

train(model)
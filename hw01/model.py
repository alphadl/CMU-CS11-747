import torch
from torch import nn
from torch.autograd import Variable

class BoW(torch.nn.Module):
    def __init__(self, nwords, ntags):
        super(BoW, self).__init__()
        self.bias = torch.zeros(ntags, requires_grad=True).type(torch.cuda.FloatTensor)
        self.emb = nn.Embedding(nwords, ntags)
        nn.init.xavier_uniform_(self.emb.weight)

    def forward(self, words):
        emb = self.emb(words)
        out = torch.sum(emb, dim=0) + self.bias  # N
        out = out.view(1, -1)  # 1 x N
        return out


class CBoW(torch.nn.Module):
    def __init__(self, nwords, ntags, emb_size):
        super(CBoW, self).__init__()

        self.W = torch.empty(ntags, emb_size, requires_grad=True).type(torch.cuda.FloatTensor)
        self.bias = torch.empty(1, ntags, requires_grad=True).type(torch.cuda.FloatTensor)
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.bias)
        
        # alternative implementation
        # self.linear = nn.Linear(emb_size, ntags)
        # nn.init.xavier_uniform_(self.linear.weight)

        self.emb = nn.Embedding(nwords, emb_size)
        nn.init.xavier_uniform_(self.emb.weight)
    
    def forward(self, words):
        emb = self.emb(words)
        out = torch.add(torch.matmul(self.W, torch.sum(emb, dim=0).view(-1, 1)).view(1, -1), self.bias)
        return out

class DeepCBoW(torch.nn.Module):
    def __init__(self, nwords, ntags, emb_size, hid_size, nlayers):
        super(DeepCBoW, self).__init__()

        self.nlayers = nlayers

        self.emb = nn.Embedding(nwords, emb_size)
        nn.init.xavier_uniform_(self.emb.weight)

        self.linears = nn.ModuleList([
            nn.Linear(emb_size if i == 0 else hid_size, hid_size) for i in range(nlayers)
        ])
        for i in range(nlayers):
            nn.init.xavier_uniform_(self.linears[i].weight)
        
        # output layer
        self.linears.append(nn.Linear(hid_size, ntags))
        nn.init.xavier_uniform_(self.linears[-1].weight)

    def forward(self, words):
        emb = self.emb(words) # nwords, emb_size
        h = torch.sum(emb, dim=0).view(1, -1) # 1, emb_size
        for i in range(self.nlayers):
            h = torch.tanh(self.linears[i](h))
        out = self.linears[-1](h)
        return out
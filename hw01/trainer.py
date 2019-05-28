from collections import defaultdict
import time
import random
import numpy as np

import torch
from torch import nn
from model import BoW, CBoW, DeepCBoW
from torch.autograd import Variable

w2i = defaultdict(lambda: len(w2i))
t2i = defaultdict(lambda: len(t2i))
UNK = w2i["<unk>"]

def read_dataset(filename):
    with open(filename, "r") as f:
        for line in f:
            tag, words = line.lower().strip().split(" ||| ")
            yield ([w2i[x] for x in words.split(" ")], t2i[tag])

# read in the data
train = list(read_dataset("../data/classes/train.txt"))
w2i = defaultdict(lambda: UNK, w2i)
dev = list(read_dataset("../data/classes/test.txt"))
nwords = len(w2i)
ntags = len(t2i)

# create the model
# BoW model
# model = BoW(nwords, ntags)
# CBoW model
# model = CBoW(nwords, ntags, emb_size=64)
# Deep CBoW model
model = DeepCBoW(nwords, ntags, emb_size=64, hid_size=64, nlayers=2)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

model = model.cuda()
print(model)

print("model parameters1:")
print(sum(p.numel() for p in model.parameters() if p.requires_grad))

# print("model parameters2:")
# pp=0
# for p in list(model.parameters()):
#     nn=1
#     print("structure:"+ str(list(p.size())))
#     print("element:",p.size())
#     for s in list(p.size()):
#         print("e:",s)
#         nn = nn*s
#     pp += nn
# print(pp)

for ITER in range(100):
    # perform training
    random.shuffle(train)
    train_loss = 0.0
    # counting the time
    start = time.time()

    for words, tag in train:
        words = torch.cuda.LongTensor(words)
        tag = torch.cuda.LongTensor([tag])
        # print("words",words)
        scores = model(words)
        loss = criterion(scores, tag)
        train_loss += loss.item()
        # print(train_loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("iter %r: train loss/sent=%.4f, time=%.2s" %
          (ITER, train_loss / len(train), time.time() - start))

    # perform validation
    test_correct = 0.0
    test_loss = 0.0
    for words, tag in dev:
        words = torch.cuda.LongTensor(words)
        tag = torch.cuda.LongTensor([tag])
        # print("tag",tag)
        scores = model(words)
        # print("score[0]",scores[0].size(),scores[0])
        loss = criterion(scores, tag)
        test_loss += loss.item()
        predict = np.argmax(scores.detach().cpu().numpy())
        # print("pred",predict)
        if predict == tag:
            test_correct += 1
    print("iter %r: test acc=%.4f, loss/sent=%.4f" % (ITER, test_correct / len(dev), test_loss/len(dev)))

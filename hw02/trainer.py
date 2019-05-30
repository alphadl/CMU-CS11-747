from collections import defaultdict
import time
import random
import numpy as np 
import math

import torch
from torch import nn
from lm_model import FFN_LM

# hyper parameters 
N = 2 # ngram
EMB_SIZE = 128
HID_SIZE = 128

w2i = defaultdict(lambda: len(w2i))
S=w2i["<s>"]
UNK=w2i["<unk>"]

def read_dataset(filename):
    with open(filename, "r") as f:
        for line in f:
            yield [w2i[x] for x in line.strip().split(" ")]

train = list(read_dataset("../data/ptb/train.txt"))
w2i = defaultdict(lambda: UNK, w2i)

dev = list(read_dataset("../data/ptb/valid.txt"))
i2w = {v:k for k, v in w2i.items()} # index 2 word
nwords = len(w2i)

# initialize the model and optimizer
model = FFN_LM(nwords=nwords, emb_size=EMB_SIZE, hid_size=HID_SIZE, num_hist=N, dropout=0.2)

model = model.cuda()
# model = nn.DataParallel(model, device_ids=[0,1])

criterion = nn.CrossEntropyLoss(size_average=sum)
optimizer = torch.optim.Adam(model.parameters())

print(model)
print("model parameters1:")
print(sum(p.numel() for p in model.parameters() if p.requires_grad))


#history score
def score_of_history(words):
    return model(torch.cuda.LongTensor(words))

#entire sentence
def loss_of_sent(sent):
    hist = [S] * N # initial hist is end symbol
    
    all_hist = []
    all_tgt = []  

    for next_word in sent + [S]:
        all_hist.append(list(hist))
        all_tgt.append(next_word)
        hist = hist[1:] + [next_word]
    
    logits = score_of_history(all_hist)
    loss = criterion(logits, torch.cuda.LongTensor(all_tgt))
    return loss

MAX_LEN = 100
#generate sentence
def gen_sent():
    hist = [S] * N
    sent = []
    while True:
        logits = score_of_history([hist])
        prob = nn.functional.softmax(logits)
        # next_word = prob.multinomial().data[0, 0]
        next_word = np.argmax(prob.detach().cpu().numpy())
        if next_word == S or len(sent) == MAX_LEN:
            break
        sent.append(next_word)
        hist=hist[1:] + [next_word]
    return sent

last_dev = 1e20
best_dev = 1e20

for ITER in range(20):
    #perform training
    random.shuffle(train)
    train_words, train_loss = 0, 0.0
    start=time.time()

    for sent_id, sent in enumerate(train):
        my_loss = loss_of_sent(sent)
        train_loss += my_loss.item()
        # train_loss += my_loss.detach()
        train_words += len(sent)

        optimizer.zero_grad()
        my_loss.backward()
        optimizer.step()

        if (sent_id+1) % 5000 ==0:
            print("--finished %r sentences (word/sec=%.2f)" % (sent_id+1, train_words/(time.time()-start)))
    print("iter %r: train loss/word=%.4f, ppl=%.4f (word/sec=%.2f)" % (ITER, train_loss/train_words, math.exp(train_loss/train_words), train_words/(time.time()-start)))

    #eval the dev set
    with torch.no_grad():
        dev_words, dev_loss = 0, 0.0
        start = time.time()
        for sent_id, sent in enumerate(dev):
            my_loss = loss_of_sent(sent)
            dev_loss += my_loss.item()
            dev_words += len(sent)
        
        
        #track the dev acc and reduce the learning rate of it
        if last_dev < dev_loss:
            optimizer.learning_rate /= 2
        last_dev = dev_loss

        #track the best dev acc and save the best model
        if best_dev > dev_loss: 
            torch.save(model, "model.pt")
            best_dev = dev_loss
        
        # Save the model
        print("iter %r: dev loss/word=%.4f, ppl=%.4f (word/sec=%.2f)" % (ITER, dev_loss/dev_words, math.exp(dev_loss/dev_words), dev_words/(time.time()-start)))
  
        # Generate a few sentences
        for _ in range(5):
            sent = gen_sent()
            print(" ".join([i2w[x] for x in sent]))
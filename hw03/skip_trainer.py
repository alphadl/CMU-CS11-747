from collections import defaultdict
import math
import time
import random
import torch
from model import cbow, skip

N = 2 # window size , gives a total 5 words: t-2 t-1 t t+1 t+2
EMB_SIZE = 128

embeddings_path = "embeddings.txt"
labels_path = "labels.txt"

w2i = defaultdict(lambda: len(w2i))
S = w2i["<s>"]
UNK = w2i["<unk>"]

def read_dataset(filename):
    with open(filename, "r") as f:
        for line in f:
            yield [w2i[x] for x in line.strip().split(" ")]

train = list(read_dataset("../data/ptb/train.txt"))
w2i = defaultdict(lambda: UNK, w2i)

dev = list(read_dataset("../data/ptb/valid.txt"))

# vocab list load finished, then generate the i2w list
i2w = {v:k for k, v in w2i.items()}

nword = len(w2i)

with open(labels_path, "w") as label_file:
    for i in range(nword):
        label_file.write(i2w[i] + '\n')

# initialize the model
# model = cbow(nword, EMB_SIZE).cuda()
model = skip(nword, EMB_SIZE).cuda()

print(model)
print("model parameters1:")
print(sum(p.numel() for p in model.parameters() if p.requires_grad))

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

print("Size of training data is %r"%len(train))
#calculate the entire sentence loss
def loss_of_sent(sent):
    #add padding to the sentence
    # padded_sent = [S] * N + sent + [S] * N
    losses = torch.cuda.FloatTensor([0])
    # losses = torch.cuda.FloatTensor(0)

    for i, word in enumerate(sent):
        for j in range(1, N+1):
            for direction in [-1, 1]:
                c = torch.cuda.LongTensor([word]) # central word
                context_id = sent[i + direction * j] if 0<= i+direction*j <len(sent) else S
                context = torch.cuda.LongTensor([context_id]) # context word
                logits = model(c)
                loss = criterion(logits, context)
                losses += loss
    # print(losses)
    return losses

MAX_LEN = 100

for ITER in range(100):
    print("started the iter %r" %ITER)

    random.shuffle(train)
    train_words, train_loss = 0, 0.0
    start = time.time()

    for sent_id, sent in enumerate(train):
        my_loss = loss_of_sent(sent)
        train_loss += my_loss.item()
        train_words += len(sent)

        optimizer.zero_grad()
        my_loss.backward()
        optimizer.step()

        if (sent_id + 1) % 5000 == 0:
            print("--finished %r sentences" % (sent_id + 1))
    
    print("iter %r: train loss/word=%.4f, ppl=%.4f, time=%.2fs" % (
    ITER, train_loss / train_words, math.exp(train_loss / train_words), time.time() - start))
    
    # Evaluate on dev set
    with torch.no_grad():
        dev_words, dev_loss = 0, 0.0
        start = time.time()

        for sent_id, sent in enumerate(dev):
            my_loss = loss_of_sent(sent)
            dev_loss += my_loss.item()
            dev_words += len(sent)
    
    print("iter %r: dev loss/word=%.4f, ppl=%.4f, time=%.2fs" % (
    ITER, dev_loss / dev_words, math.exp(dev_loss / dev_words), time.time() - start))

    print("saving embedding files")
    with open(embeddings_path, 'w') as embeddings_file:
        W_w_np = model.emb.weight.data.cpu().numpy()
        for i in range(nword):
            ith_embedding = '\t'.join(map(str, W_w_np[i]))
            embeddings_file.write(ith_embedding + '\n')
from collections import defaultdict
import math
import numpy as np
import time
import random
import torch
import torch.nn.functional as F

class skip_ns(torch.nn.Module):
    def __init__(self, nwords, emb_size):
        super(skip_ns, self).__init__()

        self.emb = torch.nn.Embedding(nwords, emb_size, sparse=True)
        torch.nn.init.xavier_uniform(self.emb.weight)

        self.context_emb = torch.nn.Embedding(nwords, emb_size, sparse=True)
        torch.nn.init.xavier_uniform(self.context_emb.weight)

    # https://arxiv.org/abs/1402.3722
    def forward(self, word_positive, context_position, negative_sample = False):
        emb = self.emb(word_positive) # 1, emb_size
        emb_context = self.context_emb(context_position) # n, emb_size
        score = torch.mm(emb_context, emb.transpose(1,0)) # n, 1

        if negative_sample:
            score = -1 * score
        
        obj = -1 * torch.sum(F.logsigmoid(score))
        return obj
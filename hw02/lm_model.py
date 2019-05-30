import torch
import torch.nn as nn

# FFN language model
class FFN_LM(nn.Module):
    def __init__(self, nwords, emb_size, hid_size, num_hist, dropout):
        super(FFN_LM, self).__init__()

        self.emb = nn.Embedding(nwords, emb_size)
        self.ffn = nn.Sequential(
            nn.Linear(num_hist * emb_size, hid_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hid_size, nwords)
        )
    
    def forward(self, words):
        emb = self.emb(words)   #[batch_size, num_hist, emb_size]
        feat = emb.view(emb.size(0), -1)   #[batch_size, num_hist * emb_size]
        logit = self.ffn(feat)   #[batch_size, nwords]

        return logit


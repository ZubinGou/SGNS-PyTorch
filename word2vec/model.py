import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class SkipGramModel(nn.Module):

    def __init__(self, emb_size, emb_dimension):
        super(SkipGramModel, self).__init__()
        self.emb_size = emb_size # v
        self.emb_dimension = emb_dimension # N
        self.u_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True) # (V, N)
        self.v_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True) # (V, N)

        initrange = 1.0 / self.emb_dimension
        init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        init.constant_(self.v_embeddings.weight.data, 0) # ? why const for v

    def forward(self, pos_u, pos_v, neg_v):
        emb_u = self.u_embeddings(pos_u) # (B, N)
        emb_v = self.v_embeddings(pos_v) # (B, N)
        emb_neg_v = self.v_embeddings(neg_v) # (B, neg_count, N)

        score = torch.sum(torch.mul(emb_u, emb_v), dim=1) # (B)
        score = torch.clamp(score, max=10, min=-10)
        score = -F.logsigmoid(score)

        neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze() # (B, neg_count)
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1) # (B)

        return torch.mean(score + neg_score) # (1)

    def save_embedding(self, id2word, file_name):
        embedding = self.u_embeddings.weight.cpu().data.numpy()
        with open(file_name, 'w', encoding='utf-8') as f:
            f.write(f"{len(id2word)} {self.emb_dimension}\n")
            for wid, w in id2word.items():
                e = ' '.join(map(lambda x: str(x), embedding[wid]))
                f.write(f"{w} {e}\n")
    







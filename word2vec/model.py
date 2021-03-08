import torch
import torch.nn as nn
import torch.nn.functional as F


class skipgram(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(skipgram, self).__init__()
        self.u_embeddings = nn.Embedding(vocab_size, embed_dim, sparse=True)
        self.v_embeddings = nn.Embedding(vocab_size, embed_dim, sparse=True)
        self.embed_dim = embed_dim

        initrange = 0.5 / self.embed_dim
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(-0, 0)  # why 0?

    def forward(self, u_pos, v_pos, v_neg):
        embed_u = self.u_embeddings(u_pos)  # (B * C * 2, N)
        embed_v = self.v_embeddings(v_pos)  # (B * C * 2, N)
        embed_neg_v = self.v_embeddings(v_neg)  # (B * C * 2, neg_sample_num, N)

        loss_pos = torch.mul(embed_u, embed_v)
        loss_neg = torch.bmm(embed_neg_v, embed_u.unsqueeze(2)).squeeze()

        loss_pos = F.logsigmoid(loss_pos.sum(1))  # (B)
        loss_neg = F.logsigmoid(-loss_neg.sum(1))  # (B)

        return -torch.mean(loss_pos + loss_neg)

    def save_embedding(self, file_name, id2word):
        embeds = self.u_embeddings.weight.cpu().data.numpy()
        with open(file_name, "w") as f:
            f.write('%d %d\n' % (len(id2word), self.embed_dim))
            for id, w in enumerate(id2word):
                e = ' '.join(map(lambda x: str(x), embeds[id]))
                f.write('%s %s\n' % (w, e))
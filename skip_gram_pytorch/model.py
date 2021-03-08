import torch
import torch.nn as nn
import torch.nn.functional as F


class skipgram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(skipgram, self).__init__()
        self.u_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        self.v_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        self.embedding_dim = embedding_dim

        initrange = 0.5 / self.embedding_dim
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(-0, 0)  # why 0?

    def forward(self, u_pos, v_pos, v_neg):
        embed_u = self.u_embeddings(u_pos)  # (B, N)
        embed_v = self.v_embeddings(v_pos)  # (B, window_size * 2, N)
        embed_neg_v = self.v_embeddings(v_neg)  # (B, window_size * 2 * neg_samplge_num, N)

        score_pos = torch.bmm(embed_v, embed_u.unsqueeze(2)).squeeze() # (B, window_size * 2)
        score_neg = torch.bmm(embed_neg_v, embed_u.unsqueeze(2)).squeeze() # (B, window_size * 2 * neg_samplge_num)

        score_pos = F.logsigmoid(score_pos.sum(1))
        score_neg = F.logsigmoid(-score_neg.sum(1))
        return -torch.mean(score_pos + score_neg)

    def input_embeddings(self):
        return self.u_embeddings.weight.data.cpu().numpy()

    def save_embedding(self, file_name, id2word):
        embeds = self.u_embeddings.weight.data
        fo = open(file_name, "w")
        for idx in range(len(embeds)):
            word = id2word(idx)
            embed = " ".join(embeds[idx])
            fo.write(word + " " + embed + "\n")

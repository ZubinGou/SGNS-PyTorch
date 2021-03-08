import torch
import torch.nn as nn
import time
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from skip_gram_pytorch.inputdata import DataReader, Word2vecDataset
from skip_gram_pytorch.utils import scorefunction
from skip_gram_pytorch.model import skipgram


class word2vec:
    def __init__(
        self,
        inputfile,
        vocabulary_size=30000,
        embedding_dim=100,
        epoch_num=10,
        batch_size=32,
        windows_size=5,
        neg_sample_num=10,
    ):
        self.data = DataReader(inputfile, vocabulary_size)
        dataset = Word2vecDataset(self.data, neg_sample_num)
        self.dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=16
        )
        self.embedding_dim = embedding_dim
        self.windows_size = windows_size
        self.vocabulary_size = vocabulary_size
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.neg_sample_num = neg_sample_num

        self.skip_gram_model = skipgram(self.vocabulary_size, self.embedding_dim)
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        if self.use_cuda:
            self.skip_gram_model.cuda()

    def train(self):
        # optimizer = optim.SGD(self.skip_gram_model.parameters(), lr=0.3)
        optimizer = optim.SparseAdam(list(self.skip_gram_model.parameters()), lr=0.001)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(self.dataloader))
        for epoch in range(1, self.epoch_num + 1):
            # print(f"\nEpoch: {epoch} ")
            start = time.time()
            self.data.process = True
            for batch_num, (pos_u, pos_v, neg_v) in enumerate(self.dataloader):
                # pos_u = pos_u.long().to(self.device)  # (B)
                # pos_v = pos_v.long().to(self.device)  # (B, window_size * 2)
                # neg_v = neg_v.long().to(
                #     self.device
                # )  # (B, window_size * 2 * neg_samplge_num)
                pos_u = Variable(pos_u.long()).to(self.device) # (B)
                pos_v = Variable(pos_v.long()).to(self.device) # (B)
                neg_v = Variable(neg_v.long()).to(self.device) # (B, neg_sample_num)
                optimizer.zero_grad()
                loss = self.skip_gram_model(pos_u, pos_v, neg_v)
                loss.backward()
                optimizer.step()

                print_per = 1000
                if batch_num % print_per == 0:
                    end = time.time()
                    word_embeddings = self.skip_gram_model.input_embeddings()
                    sp1, sp2 = scorefunction(word_embeddings)
                    print(
                        "eporch=%2d, batch=%5d: sp=%1.3f %1.3f  pair/sec = %4.2f loss=%4.3f\r"
                        % (
                            epoch,
                            batch_num,
                            sp1,
                            sp2,
                            print_per * self.batch_size / (end - start),
                            loss.data.item(),
                        ),
                        end=""
                    )
                    start = time.time()
            print()
        print("Optimization Finished!")


if __name__ == "__main__":
    wc = word2vec("text8")
    wc.train()

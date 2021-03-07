import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import time

from skip_gram_pytorch.inputdata import Options, scorefunction
from skip_gram_pytorch.model import skipgram


class word2vec:
    def __init__(self, inputfile, vocabulary_size=100000, embedding_dim=100, epoch_num=10, batch_size=32, windows_size=5, neg_sample_num=10):
        self.data = Options(inputfile, vocabulary_size)
        self.embedding_dim = embedding_dim
        self.windows_size = windows_size
        self.vocabulary_size = vocabulary_size
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.neg_sample_num = neg_sample_num

        self.skip_gram_model = skipgram(
            self.vocabulary_size, self.embedding_dim)
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        if self.use_cuda:
            self.skip_gram_model.cuda()

    def train(self):
        optimizer = optim.SGD(self.skip_gram_model.parameters(), lr=0.2)
        for epoch in range(1, self.epoch_num + 1):
            # print(f"\nEpoch: {epoch} ")
            start = time.time()
            self.data.process = True
            batch_num = 0
            batch_old = 0

            while self.data.process:
                pos_u, pos_v, neg_v = self.data.generate_batch(
                    self.windows_size, self.batch_size, self.neg_sample_num)

                pos_u = Variable(torch.LongTensor(pos_u)).to(self.device)
                pos_v = Variable(torch.LongTensor(pos_v)).to(self.device)
                neg_v = Variable(torch.LongTensor(neg_v)).to(self.device)

                optimizer.zero_grad()
                loss = self.skip_gram_modelmodel(pos_u, pos_v, neg_v, self.batch_size)
                loss.backward()
                optimizer.step()

                # if batch_num % 30000 == 0:
                #     torch.save(
                #         model.state_dict(), './tmp/skipgram.epoch{}.batch{}'.format(epoch, batch_num))

                if batch_num % 2000 == 0:
                    end = time.time()
                    word_embeddings = self.skip_gram_model.input_embeddings()
                    sp1, sp2 = scorefunction(word_embeddings)
                    print('eporch=%2d, batch=%5d: sp=%1.3f %1.3f  pair/sec = %4.2f loss=%4.3f\r'
                          % (epoch, batch_num, sp1, sp2, (batch_num-batch_old)*self.batch_size/(end-start), loss.data.item()), end="")
                    batch_old = batch_num
                    start = time.time()
                batch_num = batch_num + 1
            print()
        print("Optimization Finished!")


if __name__ == '__main__':
    wc = word2vec('text8')
    wc.train()

import torch
import time
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import optimizer

from word2vec.data_reader import DataReader
from word2vec.utils import evaluate
from word2vec.model import skipgram


class Word2VecTrainer:
    def __init__(
        self,
        inputfile,
        vocabulary_size=50000,
        embed_dim=100,
        epoch_num=10,
        batch_size=32,
        windows_size=5,
        neg_sample_num=10,
        saved_model_path="",
        output_file="sgns.vec",
    ):
        self.data = DataReader(inputfile, vocabulary_size)
        self.windows_size = windows_size
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.neg_sample_num = neg_sample_num

        self.skip_gram_model = skipgram(vocabulary_size, embed_dim)
        if saved_model_path:
            self.skip_gram_model.load_state_dict(torch.load(saved_model_path))
        self.output_file_name = output_file

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        if self.use_cuda:
            self.skip_gram_model.cuda()

    def train(self):
        # optimizer = optim.SGD(self.skip_gram_model.parameters(), lr=0.3)
        optimizer = optim.SparseAdam(list(self.skip_gram_model.parameters()), lr=0.001)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(self.dataloader))

        running_loss = 0
        for epoch in range(1, self.epoch_num + 1):
            start = time.time()
            self.data.process = True

            batch_num = 0
            running_loss = 0
            while self.data.process:
                pos_u, pos_v, neg_v = self.data.generate_batch(
                    self.windows_size, self.batch_size, self.neg_sample_num
                )
                pos_u = Variable(pos_u).to(self.device) # (B * C * 2)
                pos_v = Variable(pos_v).to(self.device) # (B * C * 2)
                neg_v = Variable(neg_v).to(self.device) # (B * C * 2, neg_sample_num)

                optimizer.zero_grad()
                loss = self.skip_gram_model(pos_u, pos_v, neg_v)
                loss.backward()
                optimizer.step()

                if batch_num % 50000 == 0:
                    torch.save(
                        self.skip_gram_model.state_dict(),
                        f"./tmp/skipgram.epoch{epoch}.batch{batch_num}",
                    )
                    self.skip_gram_model.save_embedding(f"tmp/epoch{epoch}.batch{batch_num}.vec", self.data.id2word)

                running_loss = running_loss * 0.99 + loss.data.item() * 0.01
                print_per = 10000
                if batch_num % print_per == 0:
                    end = time.time()
                    word_embeddings = self.skip_gram_model.u_embeddings.weight.cpu().data.numpy()
                    sp1, sp2 = evaluate(word_embeddings, self.data.word2id)
                    print(
                        "epoch=%2d, batch=%5d: sp=%1.3f %1.3f  pair/sec = %4.2f loss=%4.3f" % (
                            epoch,
                            batch_num,
                            sp1,
                            sp2,
                            print_per * self.batch_size / (end - start),
                            running_loss,
                        ),
                        end="\n",
                    )
                    start = time.time()
                batch_num = batch_num + 1

            self.skip_gram_model.save_embedding(self.output_file_name, self.data.id2word)
            print()
        print("Optimization Finished!")


if __name__ == "__main__":
    wc = Word2VecTrainer("text8")
    wc.train()

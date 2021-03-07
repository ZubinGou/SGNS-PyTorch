import torch
import torch.optim as optim
from torch.optim import optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from word2vec.data_reader import DataReader, Word2vecDataset
from word2vec.model import SkipGramModel


class Word2VecTrainer:
    def __init__(self, input_file, output_file, emb_dimension=100, batch_size=32, window_size=5, epochs=3,
                 initial_lr=0.001, min_count=12):

        self.data = DataReader(input_file, min_count)
        dataset = Word2vecDataset(self.data, window_size)
        self.dataloader = DataLoader(dataset, batch_size=batch_size,
                                     shuffle=False, num_workers=0, collate_fn=dataset.collate)

        self.output_file_name = output_file
        self.vocab_size = len(self.data.word2id)
        self.emb_dimension = emb_dimension
        self.batch_size = batch_size
        self.epochs = epochs
        self.initial_lr = initial_lr
        self.skip_gram_model = SkipGramModel(self.vocab_size, self.emb_dimension)

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")


    def train(self):

        # optimizer = optim.SparseAdam(self.skip_gram_model.parameters(), lr=self.initial_lr) # ? list()
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(self.dataloader))
        optimizer = optim.SGD(self.skip_gram_model.parameters(), lr=0.2)
        for epoch in range(self.epochs):

            print("\nEpoch: " + str(epoch + 1))

            running_loss = 0.0
            for i, sample_batched in enumerate(tqdm(self.dataloader)):

                if len(sample_batched[0]) > 1:
                    pos_u = sample_batched[0].to(self.device) # (B, 1)
                    pos_v = sample_batched[1].to(self.device) # (B, 1)
                    neg_v = sample_batched[2].to(self.device) # (B, neg_count)

                    # scheduler.step()
                    optimizer.zero_grad()
                    loss = self.skip_gram_model(pos_u, pos_v, neg_v)
                    loss.backward()
                    optimizer.step()

                    # running_loss = running_loss * 0.9 + loss.item() * 0.1
                    running_loss = loss.item()

                    if i > 0 and i % 2 == 0:
                        print(" Loss: " + str(running_loss))

            self.skip_gram_model.save_embedding(self.data.id2word, self.output_file_name)


if __name__ == '__main__':
    from gensim.test.utils import datapath
    corpus_path = datapath('lee_background.cor')
    w2v = Word2VecTrainer(input_file=corpus_path, output_file="out.vec")
    w2v.train()

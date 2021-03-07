import numpy as np
from numpy.core.fromnumeric import reshape
import torch
from torch.utils.data import Dataset 

np.random.seed(114514)


class DataReader:
    NEGTIVE_TABLE_SIZE = 1e8

    def __init__(self, input_file_name, min_count):

        self.negatives = []  # negative sampling
        self.discards = []  # subsampling
        self.negpos = 0

        self.word2id = dict()
        self.id2word = dict()
        self.sentences_count = 0
        self.token_count = 0
        self.word_frequency = dict()

        self.input_file_name = input_file_name
        self.read_words(min_count)
        self.init_table_negatives()
        self.init_table_discards()

    def read_words(self, min_count):
        word_frequency = dict()
        for line in open(self.input_file_name, encoding="utf-8"):
            line = line.split()
            if len(line) > 1:
                self.sentences_count += 1
                for word in line:
                    self.token_count += 1
                    word_frequency[word] = word_frequency.get(word, 0) + 1

                    if self.token_count % 1000000 == 0:
                        print(f"Read {int(self.token_count / 1000000)} M words.")

        wid = 0
        for w, c in word_frequency.items():
            if c < min_count:
                continue
            self.word2id[w] = wid
            self.id2word[wid] = w
            self.word_frequency[wid] = c
            wid += 1
        print(f"Total embeddings: {len(self.word2id)}")

    def init_table_discards(self):  # subsampling
        t = 0.0001
        f = np.array(list(self.word_frequency.values())) / self.token_count
        self.discards = np.sqrt(t / f) + (t / f)

    def init_table_negatives(self):
        pow_frequency = np.array(list(self.word_frequency.values())) ** 0.75  # 0.5(from fasttext) or 0.75 not matter too much.
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = np.round(ratio * DataReader.NEGTIVE_TABLE_SIZE)
        for wid, c in enumerate(count):
            self.negatives += [wid] * int(c)
        self.negatives = np.array(self.negatives)
        np.random.shuffle(self.negatives)

    def get_negatives(self, target, size):
        negs = self.negatives[self.negpos : self.negpos + size]
        self.negpos = (self.negpos + size) % len(self.negatives)
        if len(negs) != size:
            return np.concatenate((negs, self.negatives[0 : self.negpos]))
        # check equality with target
        for i in range(len(negs)):
            if negs[i] == target:
                negs[i] = self.negatives[self.negpos]
                self.negpos = (self.negpos + 1) % len(self.negatives)
        return negs


# ---------------------------------------------------------------------------------------------------


class Word2vecDataset(Dataset):
    def __init__(self, data, window_size):
        self.data = data
        self.window_size = window_size
        self.input_file = open(data.input_file_name, encoding="utf-8")

    def __len__(self):
        return self.data.sentences_count

    def __getitem__(self, idx):
        while True:
            line = self.input_file.readline()
            if not line:  # 循环读？
                self.input_file.seek(0, 0)
                line = self.input_file.readline()

            if len(line) > 1:
                words = line.split()

                if len(words) > 1:
                    word_ids = [
                        self.data.word2id[w]
                        for w in words
                        if w in self.data.word2id
                        and np.random.rand() < self.data.discards[self.data.word2id[w]]
                    ]
                    boundary = np.random.randint(
                        1, self.window_size + 1
                    )  # ? window_size + 1
                    return [
                        (u, v, self.data.get_negatives(v, 5))
                        for i, u in enumerate(word_ids)
                        for v in word_ids[max(i - boundary, 0) : i + boundary]
                        if u != v
                    ]

    @staticmethod
    def collate(batches): # ? 
        all_u = [u for batch in batches for u, _, _ in batch if len(batch) > 0]
        all_v = [v for batch in batches for _, v, _ in batch if len(batch) > 0]
        all_neg_v = [neg_v for batch in batches for _, _, neg_v in batch if len(batch) > 0]

        return torch.LongTensor(all_u), torch.LongTensor(all_v), torch.LongTensor(all_neg_v)
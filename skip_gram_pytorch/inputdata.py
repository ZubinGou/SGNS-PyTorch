import os
import random
import collections
import numpy as np

from torch.utils.data import Dataset

data_index = 0


class DataReader(object):
    NEGATIVE_TABLE_SIZE = 1e8

    def __init__(self, datafile, vocabulary_size):
        self.text_size = vocabulary_size
        self.save_path = "tmp"
        self.negatives = []
        self.negpos = 0  # used for negtive sampling

        self.text = self.read_data(datafile)
        self.build_dataset(self.text, self.text_size)
        self.save_vocab()
        self.train_data = self.subsampling(self.data_encoded)
        self.init_negtives_table()


    def read_data(self, filename):
        with open(filename) as f:
            data = f.read().split()
            print("len data:", len(data))
            data = [x.lower() for x in data]
        return data

    def build_dataset(self, words, n_words):
        counter = dict(collections.Counter(words).most_common(n_words - 1))
        counter["<unk>"] = len(words) - np.sum(list(counter.values()))

        self.id2word = [word for word in counter.keys()]
        self.word2id = {word: id for id, word in enumerate(self.id2word)}

        self.word_count = np.array(list(counter.values()), dtype=np.float32)
        self.word_frequency = self.word_count / np.sum(self.word_count)
        self.data_encoded = [self.word2id[word] for word in words]

    def save_vocab(self):
        with open(os.path.join(self.save_path, "vocab.txt"), "w") as f:
            for i in range(len(self.word_count)):
                vocab_word = self.id2word[i]
                f.write("%s %d\n" % (vocab_word, self.word_count[i]))

    def init_negtives_table(self):
        pow_frequency = self.word_frequency ** 0.75
        ratio = pow_frequency / np.sum(pow_frequency)
        count = np.round(ratio * self.NEGATIVE_TABLE_SIZE)
        for wid, c in enumerate(count):
            self.negatives += [wid] * int(c)
        self.negatives = np.array(self.negatives)
        np.random.shuffle(self.negatives)

    def subsampling(self, data):
        t = 0.0001
        discards = np.sqrt(t / self.word_frequency) + (t / self.word_frequency)
        subsampled_data = [id for id in data if random.random() < discards[id]]
        return subsampled_data

    def generate_batch(self, window_size, batch_size, count):
        data = self.train_data
        global data_index
        span = 2 * window_size + 1
        context = np.ndarray(shape=(batch_size, 2 * window_size), dtype=np.int64)
        labels = np.ndarray(shape=(batch_size), dtype=np.int64)
        pos_pair = []

        if data_index + span > len(data):
            data_index = 0
            self.process = False
        buffer = data[data_index : data_index + span]
        pos_u = []
        pos_v = []

        for i in range(batch_size):
            data_index += 1
            context[i, :] = buffer[:window_size] + buffer[window_size + 1 :]
            labels[i] = buffer[window_size]
            if data_index + span > len(data):
                buffer[:] = data[:span]
                data_index = 0
                self.process = False
            else:
                buffer = data[data_index : data_index + span]

            for j in range(span - 1):
                pos_u.append(labels[i])
                pos_v.append(context[i, j])
        neg_v = np.random.choice(
            self.negatives, size=(batch_size * 2 * window_size, count)
        )
        return np.array(pos_u), np.array(pos_v), neg_v

    def get_negatives(self, size):
        negs = self.negatives[self.negpos : self.negpos + size]
        self.negpos = (self.negpos + size) % len(self.negatives)
        if len(negs) != size:
            return np.concatenate((negs, self.negatives[0 : self.negpos]))
        # check equality with target
        # for i in range(len(negs)):
        #     if negs[i] == target:
        #         negs[i] = self.negatives[self.negpos]
        #         self.negpos = (self.negpos + 1) % len(self.negatives)
        return negs


class Word2vecDataset(Dataset):
    def __init__(self, data, window_size, neg_count=10):
        self.data = data
        self.window_size = window_size
        self.neg_count = neg_count  # The paper says that selecting 5-20 words works well for smaller datasets, and you can get away with only 2-5 words for large datasets.

    def __len__(self):
        return len(self.data.train_data)

    def __getitem__(self, idx):
        pos_u = np.array(self.data[idx])

        C = self.window_size
        pos_v_idx = list(range(max(idx - C, 0), idx)) + list(
            range(idx + 1, min(idx + C + 1, len(self.data) - 1))
        )
        pos_v = np.array(self.data[pos_v_idx])
        neg_v = self.data.get_negatives(2 * self.window_size * self.neg_count)
        return pos_u, pos_v, neg_v
        # neg_v = np.random.choice(self.data.negatives, size=2*self.window_size*self.neg_count)

import os
import random
import collections
import numpy as np
import torch

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
        self.data_encoded = [
            self.word2id.get(word, self.word2id["<unk>"]) for word in words
        ]

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

    def generate_batch(self, window_size, batch_size, neg_sample_num):
        data = self.train_data
        global data_index
        span = 2 * window_size + 1

        pos_u = []  # (B * window_size * 2)
        pos_v = []  # (B * window_size * 2)

        for _ in range(batch_size):
            if data_index + span > len(data):
                buffer = data[:span]
                data_index = 0
                self.process = False
            else:
                buffer = data[data_index : data_index + span]

            pos_u.extend([buffer[window_size]] * window_size * 2)
            pos_v.extend(buffer[:window_size] + buffer[window_size + 1 :])

            data_index += 1

        neg_v = np.random.choice(
            self.negatives, size=(batch_size * 2 * window_size, neg_sample_num)
        )
        return torch.LongTensor(pos_u), torch.LongTensor(pos_v), torch.LongTensor(neg_v)
import csv
import json
from scipy.stats import spearmanr
import collections
import numpy as np

import math
import os
import random

data_index = 0


class Options(object):
    def __init__(self, datafile, vocabulary_size):
        self.vocabulary_size = vocabulary_size
        self.save_path = "tmp"
        self.vocabulary = self.read_data(datafile)
        data_or, self.count, self.vocab_words = self.build_dataset(self.vocabulary,
                                                                   self.vocabulary_size)
        self.train_data = self.subsampling(data_or)
        #self.train_data = data_or

        self.sample_table = self.init_sample_table()

        self.save_vocab()

    def read_data(self, filename):
        with open(filename) as f:
            data = f.read().split()
            # data = [x for x in data if x != 'eoood']
        return data

    def build_dataset(self, words, n_words):
        """Process raw inputs into a ."""
        count = [['UNK', -1]]
        count.extend(collections.Counter(words).most_common(n_words - 1))
        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary)
        data = list()
        unk_count = 0
        for word in words:
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0  # dictionary['UNK']
                unk_count += 1
            data.append(index)
        count[0][1] = unk_count
        reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        return data, count, reversed_dictionary

    def save_vocab(self):
        with open(os.path.join(self.save_path, "vocab.txt"), "w") as f:
            for i in xrange(len(self.count)):
                vocab_word = self.vocab_words[i]
                f.write("%s %d\n" % (vocab_word, self.count[i][1]))

    def init_sample_table(self):
        count = [ele[1] for ele in self.count]
        pow_frequency = np.array(count)**0.75
        power = sum(pow_frequency)
        ratio = pow_frequency / power
        table_size = 1e8
        count = np.round(ratio*table_size)
        sample_table = []
        for idx, x in enumerate(count):
            sample_table += [idx]*int(x)
        return np.array(sample_table)

    def weight_table(self):
        count = [ele[1] for ele in self.count]
        pow_frequency = np.array(count)**0.75
        power = sum(pow_frequency)
        ratio = pow_frequency / power
        return np.array(ratio)

    def subsampling(self, data):
        count = [ele[1] for ele in self.count]
        frequency = np.array(count)/sum(count)
        P = dict()
        for idx, x in enumerate(frequency):
            y = (math.sqrt(x/0.001)+1)*0.001/x
            P[idx] = y
        subsampled_data = list()
        for word in data:
            if random.random() < P[word]:
                subsampled_data.append(word)
        return subsampled_data

    def generate_batch(self, window_size, batch_size, count):
        data = self.train_data
        global data_index
        span = 2 * window_size + 1
        context = np.ndarray(
            shape=(batch_size, 2 * window_size), dtype=np.int64)
        labels = np.ndarray(shape=(batch_size), dtype=np.int64)
        pos_pair = []

        if data_index + span > len(data):
            data_index = 0
            self.process = False
        buffer = data[data_index:data_index + span]
        pos_u = []
        pos_v = []

        for i in range(batch_size):
            data_index += 1
            context[i, :] = buffer[:window_size]+buffer[window_size+1:]
            labels[i] = buffer[window_size]
            if data_index + span > len(data):
                buffer[:] = data[:span]
                data_index = 0
                self.process = False
            else:
                buffer = data[data_index:data_index + span]

            for j in range(span-1):
                pos_u.append(labels[i])
                pos_v.append(context[i, j])
        neg_v = np.random.choice(self.sample_table, size=(
            batch_size*2*window_size, count))
        return np.array(pos_u), np.array(pos_v), neg_v


def cosine_similarity(v1, v2):
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]
        y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return sumxy/math.sqrt(sumxx*sumyy)


def scorefunction(embed):
    f = open('./tmp/vocab.txt')
    line = f.readline()
    vocab = []
    wordindex = dict()
    index = 0
    while line:
        word = line.strip().split()[0]
        wordindex[word] = index
        index = index + 1
        line = f.readline()
    f.close()
    ze = []
    with open('./skip_gram_pytorch/wordsim353/combined.csv') as csvfile:
        filein = csv.reader(csvfile)
        index = 0
        consim = []
        humansim = []
        for eles in filein:
            if index == 0:
                index = 1
                continue
            if (eles[0] not in wordindex) or (eles[1] not in wordindex):
                continue

            word1 = int(wordindex[eles[0]])
            word2 = int(wordindex[eles[1]])
            humansim.append(float(eles[2]))

            value1 = embed[word1]
            value2 = embed[word2]
            index = index + 1
            score = cosine_similarity(value1, value2)
            consim.append(score)

    cor1, pvalue1 = spearmanr(humansim, consim)

    if 1 == 1:
        lines = open('./skip_gram_pytorch/rw/rw.txt', 'r').readlines()
        index = 0
        consim = []
        humansim = []
        for line in lines:
            eles = line.strip().split()
            if (eles[0] not in wordindex) or (eles[1] not in wordindex):
                continue
            word1 = int(wordindex[eles[0]])
            word2 = int(wordindex[eles[1]])
            humansim.append(float(eles[2]))

            value1 = embed[word1]
            value2 = embed[word2]
            index = index + 1
            score = cosine_similarity(value1, value2)
            consim.append(score)

    cor2, pvalue2 = spearmanr(humansim, consim)

    return cor1, cor2

import math
import csv
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from gensim.test.utils import datapath

def evaluate(embed, word2id):

    def get_correlation(embed, word2id, file_name):
        human_sim = []
        wv_sim = []
        with open(file_name, "r") as f:
            lines = f.readlines()
            for line in lines:
                if line[0] == '#':
                    continue
                line = line.strip().split()
                if (line[0] not in word2id) or (line[1] not in word2id):
                    continue
                word1 = int(word2id[line[0]])
                word2 = int(word2id[line[1]])
                human_sim.append(float(line[2]))

                score = cosine_similarity(embed[word1].reshape(1, -1), embed[word2].reshape(1, -1))[0][0]
                wv_sim.append(score)

        cor, _ = spearmanr(human_sim, wv_sim)
        return cor

    path_ws353 = datapath('wordsim353.tsv')
    path_rw = './test_data/rw_clean.txt'
    path_sl999 = datapath('simlex999.txt')

    cor_ws353 = get_correlation(embed, word2id, path_ws353)
    cor_rw = get_correlation(embed, word2id, path_rw)
    cor_sl999 = get_correlation(embed, word2id, path_sl999)

    return cor_ws353, cor_rw, cor_sl999
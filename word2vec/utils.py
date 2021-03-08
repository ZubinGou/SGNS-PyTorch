import math
import csv
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity


# def cosine_similarity(v1, v2):
#     "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
#     sumxx, sumxy, sumyy = 0, 0, 0
#     for i in range(len(v1)):
#         x = v1[i]
#         y = v2[i]
#         sumxx += x * x
#         sumyy += y * y
#         sumxy += x * y
#     return sumxy / math.sqrt(sumxx * sumyy)


def evaluate(embed, word2id):
    human_sim = []
    wv_sim = []
    with open("./word2vec/wordsim353/combined.csv") as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        for line in reader:
            if (line[0] not in word2id) or (line[1] not in word2id):
                continue

            word1 = int(word2id[line[0]])
            word2 = int(word2id[line[1]])
            human_sim.append(float(line[2]))

            score = cosine_similarity(embed[word1].reshape(1, -1), embed[word2].reshape(1, -1))[0][0]
            wv_sim.append(score)

    cor1, _ = spearmanr(human_sim, wv_sim)

    with open("./word2vec/rw/rw.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            if (line[0] not in word2id) or (line[1] not in word2id):
                continue
            word1 = int(word2id[line[0]])
            word2 = int(word2id[line[1]])
            human_sim.append(float(line[2]))

            score = cosine_similarity(embed[word1].reshape(1, -1), embed[word2].reshape(1, -1))[0][0]
            wv_sim.append(score)

    cor2, _ = spearmanr(human_sim, wv_sim)

    return cor1, cor2
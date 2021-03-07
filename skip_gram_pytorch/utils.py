import math
import csv
from scipy.stats import spearmanr


def cosine_similarity(v1, v2):
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]
        y = v2[i]
        sumxx += x * x
        sumyy += y * y
        sumxy += x * y
    return sumxy / math.sqrt(sumxx * sumyy)


def scorefunction(embed):
    f = open("./tmp/vocab.txt")
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
    with open("./skip_gram_pytorch/wordsim353/combined.csv") as csvfile:
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
        lines = open("./skip_gram_pytorch/rw/rw.txt", "r").readlines()
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
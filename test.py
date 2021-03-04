import scipy
import sklearn
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

from word2vec.trainer import Word2VecTrainer

def evaluate(filename, embedding_weights):
    if filename.endswith('.csv'):
        data = pd.read_csv(filename, seq=',')
    else:
        data = pd.read_csv(filename, seq='\t')
    
    human_similarity = []
    model_similarity = []
    for i in data.iloc[i, 0:2].index:  # data.iloc[:, 0:2]取所有行索引为0、1的数据
        word1 , word2 = data.iloc[i, 0], data.iloc[i, 1]
        if word1 not in word2id or word2 not in word2id:
            continue
        else:
            word1_idx, word2_idx = word2id[word1], word2id[word2]
            word1_embed, word2_embed = embedding_weights[[word1_idx]], embedding_weights[[word2_idx]]
            # 模型计算的相似度
            model_similarity.append(float(sklearn.metrics.pairwise.cosine_similarity(word1_embed, word2_embed)))
             # 已知的相似度
            human_similarity.append(float(data.iloc[i, 2]))
    
    # 两者相似度的差异性
    return scipy.stats.spearmanr(human_similarity, model_similarity)

# 取cos 最近的十个单词
def find_nearest(word, embedding_weights):
    index = word2id[word]
    embedding = embedding_weights[index]
    cos_dis = np.array([scipy.spatial.distance.cosine(e, embedding) for e in embedding_weights])
    return [id2word[i] for i in cos_dis.argsort()[:10]]

if __name__ == "__main__":
    w2v = Word2VecTrainer(input_file="input.txt", output_file="out.vec")
    w2v.train()

    word2id = w2v.data.word2id
    id2word = w2v.data.id2word

    # word similarity
    embedding_weights = w2v.skip_gram_model.u_embeddings #(V, N)
    print("simlex-999", evaluate("simlex-999.txt", embedding_weights))
    print("men", evaluate("men.txt", embedding_weights))
    print("wordsim353", evaluate("wordsim353.csv", embedding_weights))

    # nearest words
    for word in ["good", "fresh", "monster", "green", "like", "america", "chicago", "work", "computer", "language"]:
        print(word, find_nearest(word))
    
    # nearest words: euclidean
    
    distance_matrix = euclidean_distances(embedding_weights)
    print(distance_matrix.shape)
    similar_words = {search_term: [id2word[idx] for idx in distance_matrix[word2id[search_term]-1].argsort()[1:6]+1] 
                    for search_term in ["good", "fresh", "monster", "green", "like", "america", "chicago", "work", "computer", "language"]}

    # word relation
    man_idx = word2id["man"] 
    king_idx = word2id["king"] 
    woman_idx = word2id["woman"]
    embedding = embedding_weights[woman_idx] - embedding_weights[man_idx] + embedding_weights[king_idx]
    cos_dis = np.array([scipy.spatial.distance.cosine(e, embedding) for e in embedding_weights])
    for i in cos_dis.argsort()[:20]:
        print(id2word[i])
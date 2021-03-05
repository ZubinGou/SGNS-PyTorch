import gensim.models.keyedvectors as word2vec
import gensim.downloader as api


def load_word2vec():
    """ Load Word2Vec Vectors
        Return:
            wv_from_bin: All 3 million embeddings, each lengh 300
    """
    # wv_from_bin = api.load("word2vec-google-news-300")
    wv_from_bin = word2vec.KeyedVectors.load_word2vec_format('dataset/GoogleNews-vectors-negative300.bin', binary=True)
    vocab = list(wv_from_bin.vocab.keys())
    print("Loaded vocab size %i" % len(vocab))
    return wv_from_bin

if __name__ == "__main__":
    wv = load_word2vec()
    for index, word in enumerate(wv.index_to_key):
        if index == 10:
            break
        print(f"word #{index}/{len(wv.index_to_key)} is {word}")
    print(wv)
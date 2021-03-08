from gensim.test.utils import datapath
from gensim import utils
from gensim.models import Word2Vec
import gensim.downloader as api
import pprint
import tqdm

from skip_gram_pytorch.word2vec2 import word2vec
pp = pprint.PrettyPrinter()

wv = word2vec("dataset/text8.txt") # emb_dim=100, vocab=30000, SparseAdam, lr=0.001
# wv = word2vec("dataset/text8.txt")
wv.train()
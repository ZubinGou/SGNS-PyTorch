from gensim.test.utils import datapath
from gensim.models import KeyedVectors

def evaluate_wordsim(wordvec):
    p_ws353 = wordvec.evaluate_word_pairs(datapath('wordsim353.tsv'))[1][0]
    p_rw = wordvec.evaluate_word_pairs("word2vec/rw/rw_clean.txt")[1][0]
    p_sl999 = wordvec.evaluate_word_pairs(datapath('simlex999.txt'))[1][0]
    print("WS353:", p_ws353)
    print("RW:", p_rw)
    print("SL999", p_sl999)

wv = KeyedVectors.load_word2vec_format("sgns.vec", binary=False)
# wv = KeyedVectors.load_word2vec_format("tmp/epoch2.batch200000.vec", binary=False)

print("Loaded vocab size %i" % len((wv.vocab)))
evaluate_wordsim(wv)
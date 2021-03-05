# SGNS-PyTorch
SkipGram NegativeSampling implemented in PyTorch.


## Paper
1. [Efficient Estimation of Word Representations in Vector Space](http://arxiv.org/pdf/1301.3781.pdf) (original word2vec paper)
2. [Distributed Representations of Words and Phrases and their Compositionality](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) (negative sampling paper)

## Notes
![](images/skip-gram.png)

Word2Vec是用无监督方式从文本中学习词向量来表征语义信息的模型，语义相近的词在嵌入空间中距离相近。类似于auto-encoder，Word2Vec训练的神经网络不用于处理新任务，真正需要的是模型参数，即隐层的权重矩阵。

Skip-gram是在给定目标单词的情况下，预测其上下文单词。

用两个word matrix，W表示目标单词向量矩阵(V\*N)，W'表示上下文单词向量矩阵（N\*V），词向量维度N，词汇表维度V。

模型：
1. 投影：$h_i=Wx_k$
2. 计算相似度：$z=W'h_i$
3. 转换为概率分布：$\hat y=\text{softmax}(z)$

高效训练的三个trick（来自第二篇paper）：
1. subsampling of the frequent words
2. nagative sampling (alternative to hierarchical softmax)
3. treat word pairs / phases as one word

### Subsampling
高频词数量远超训练所需，所以进行抽样，基于词频以一定概率丢弃词汇（论文中公式）：
$$
P\left(w_{i}\right)=1-\sqrt{\frac{t}{f\left(w_{i}\right)}}
$$

作者实际使用的公式（t默认0.0001）：
$$P\left(w_{i}\right)=\sqrt{\frac{t}{f\left(w_{i}\right)}} + \frac{t}{f\left(w_{i}\right)}$$

### Negative Sampling
负采样使得每个训练样本仅更新一小部分权重。negative word指期望概率为0的单词，选取概率为：
$$
P_n(w_i)=f(w_i)^{3 / 4} / Z
$$

## 评估
1. 基于词向量的语言学特性
    - similarity task 词相似
    - analogy task 词类比 (A-B=C-D)
2. Task-specific
    - 对具体任务的性能提升

这里我们基于词相似，在 `WordSim-353` and `SimLex-999`上进行评估

语料：[One Billion Words Benchmark for Language Modeling](https://opensource.google/projects/lm-benchmark)


## Reference
- https://github.com/theeluwin/pytorch-sgns
- https://github.com/fanglanting/skip-gram-pytorch
- https://github.com/Andras7/word2vec-pytorch
- https://www.kaggle.com/karthur10/skip-gram-implementation-with-pytorch-step-by-step

## TODO
- One Billion Word Benchmark
- Train and test
- <unk>
# SPPMI-SVD
## Paper: [Neural Word Embedding as Implicit Matrix Factorization](https://papers.nips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization.pdf)
## Authors: Omer Levy, and Yoav Goldberg
## My explanation of this paper [link](https://github.com/a1da4/paper/issues/27)
## Arguments
* --file\_path: path, corpus you want to train
* --id2word\_path: path, index to word file path (split by tab)  
I will fix this path, not text file but dict type pickle
* --threshold: int, adopt threshold to cooccur matrix or not
* --has\_abs\_dis: bool(call this argument: True, else False), adopt absolute discoutning smoothing or not
* --window\_size: int, window size in counting co-occurence
* --w2v\_sgns: int, num of negative samples in word2vec (in SPPMI-SVD, SPPMI uses -log(#negative samples) )
* --wv\_size: int, size of word vector  
I will fix this path into '--size'

# SPPMI-SVD
## Paper: [Neural Word Embedding as Implicit Matrix Factorization](https://papers.nips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization.pdf)
- Authors: Omer Levy, and Yoav Goldberg  
- NIPS 2014  
- My literature review is here [link](https://github.com/a1da4/paper/issues/27)  
## Arguments
* -f, --file\_path: path, corpus you want to train
* -p, --pickle\_id2word: path, pickle of index2word dictionary
* -t, --threshold: int, adopt threshold to cooccur matrix or not
* -a, --has\_abs\_dis: bool(call this argument: True, else False), adopt absolute discoutning smoothing or not
* -c, --has\_cds: bool(call this argument: True, else False), adopt contextual distribution smoothing or not
* -w, --window\_size: int, window size in counting co-occurence
* -s, --shift: int, num of negative samples in word2vec (in SPPMI-SVD, SPPMI uses -log(#negative samples) )
* -d, --dim: int, size of word vector  

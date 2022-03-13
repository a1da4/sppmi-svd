# SPPMI-SVD
## Paper: [Neural Word Embedding as Implicit Matrix Factorization](https://papers.nips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization.pdf)
- Authors: Omer Levy, and Yoav Goldberg  
- NIPS 2014  
- My literature review is here [link](https://github.com/a1da4/paper/issues/27)  

## Preprocess  
1. `subsample.py` (if you need): remove words with a frequency-based probability to mitigate the effects of high-frequency words.
   - -f, --file\_path: path, corpus
   - -t, --threshold: float, threshold of remove probability (default is 1e-3)
2. `make\_id2word.py`: obtain target words from corpus.
   - -f, --file\_path: path, corpus
   - -t, --threshold: int, threshold of target words

## Train SPPMI-SVD (`main.py`) 
### Arguments
- -f, --file\_path: path, corpus you want to train
- -p, --pickle\_id2word: path, pickle of index2word dictionary
- --cooccur\_pretrained: path, output text file of pre-trained co-occur matrix
- --sppmi\_pretrained: path, output text file of pre-trained sppmi matrix
- -t, --threshold: int, adopt threshold to cooccur matrix or not
- -a, --has\_abs\_dis: bool(call this argument: True, else False), adopt absolute discoutning smoothing or not
<img width="528" alt="スクリーンショット 2020-03-16 2 28 25" src="https://user-images.githubusercontent.com/45454055/76706835-d1d7ee00-672d-11ea-9d31-0b83d4dbeb79.png">

- -c, --has\_cds: bool(call this argument: True, else False), adopt contextual distribution smoothing or not
![スクリーンショット 2020-08-12 23 51 15](https://user-images.githubusercontent.com/45454055/90030195-b8e06280-dcf6-11ea-9aa0-c21055fa44fc.png)

- -w, --window\_size: int, window size in counting co-occurence
- -s, --shift: int, num of negative samples in word2vec (in SPPMI-SVD, SPPMI uses -log(#negative samples) )
- -d, --dim: int, size of word vector

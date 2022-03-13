# SPPMI-SVD
## Paper: [Neural Word Embedding as Implicit Matrix Factorization](https://papers.nips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization.pdf)
- Authors: Omer Levy, and Yoav Goldberg  
- NIPS 2014  
- My literature review is here [link](https://github.com/a1da4/paper/issues/27)  

## Preprocess  
1. `subsample.py` (if you need): remove words with a frequency-based probability to mitigate the effects of high-frequency words.
   - -f, --file\_path: path, corpus
   - -t, --threshold: float, threshold of remove probability (default is 1e-3)
   ![スクリーンショット 2022-03-13 17 32 56](https://user-images.githubusercontent.com/45454055/158051633-a26ef501-2509-4c71-bf1e-2166773d4ad1.png)  
   many papers assign t to 1e-5, but in this case (_kokoro_, Soseki Natsume, 6383 types, 102763 tokens), **87% of words are removed.**
   ```
   INFO:root:[main] args: Namespace(file_path='kokoro_processed.txt', threshold=1e-05)
   INFO:root:[main] Count (raw) word frequency...
   INFO:root:[main] - most frequent words: [('の', 5818), ('た', 5350), ('。', 4654), ('に', 4363), ('は', 4037)]
   INFO:root:[main] - total_freq: 102763
   INFO:root:[main] Subsampling...
   INFO:root:[main] - remove probability of most frequent words: [('の', 0.9867097996283141), ('た', 0.9861406936020674), ('。', 0.9851404657378731), ('に', 0.9846529191631387), ('は', 0.984045286407889)]
   INFO:root:[main] Save processed document...
   INFO:root:[main] Count (subsampled) word frequency...
   INFO:root:[main] - most frequent words: [('た', 83), ('、', 76), ('は', 71), ('。', 71), ('て', 69)]
   INFO:root:[main] - total_freq: 13380
   INFO:root:[main] - 0.869797495207419% words are removed
   ```
   When we assign t to 0.01, 21% of words are removed (just right?).
   ```
   INFO:root:[main] args: Namespace(file_path='kokoro_processed.txt', threshold=0.01)
   INFO:root:[main] Count (raw) word frequency...
   INFO:root:[main] - most frequent words: [('の', 5818), ('た', 5350), ('。', 4654), ('に', 4363), ('は', 4037)]
   INFO:root:[main] - total_freq: 102763
   INFO:root:[main] Subsampling...
   INFO:root:[main] - remove probability of most frequent words: [('の', 0.5797269626545619), ('た', 0.5617302499238903), ('。', 0.5301002676236953), ('に', 0.5146826912079521), ('は', 0.4954676563328255)]
   INFO:root:[main] Save processed document...
   INFO:root:[main] Count (subsampled) word frequency...
   INFO:root:[main] - most frequent words: [('の', 2401), ('た', 2277), ('。', 2181), ('に', 2101), ('は', 2051)]
   INFO:root:[main] - total_freq: 81176
   INFO:root:[main] - 0.21006587974270896% words are removed
   ```
   From these results, **we need to tune this paramter t.** 
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

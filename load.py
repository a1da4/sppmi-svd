# main.py で学習したモデルを用いて、複数の埋め込み空間に共起する単語の距離を計算する
import sys
import numpy as np
from util import most_similar

wordvec_size=100

f_ = sys.argv
model_name = f_[-1]

word_vec_svd = np.load(model_name)
word_vec_svd = word_vec_svd[:, :wordvec_size]

with open("id_to_word.txt") as f:
    pairs = f.readlines()
    word_to_id = {}
    id_to_word = {}
    import re
    for p in pairs:
        p = re.sub(r"\n", "", p)
        p = p.split("\t")
        id = int(p[0])
        word = p[1]
        id_to_word[id] = word
        word_to_id[word] = id
#print(word_to_id, id_to_word)
for word in word_to_id:
    #print(most_similar(word, word_to_id, id_to_word, word_vec_svd))
    most_similar(word, word_to_id, id_to_word, word_vec_svd)

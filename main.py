import sys
import numpy as np
import re
from util import preprocess, create_co_matrix, sppmi, most_similar

f_name = sys.argv
f_name = f_name[1]
texts = []
with open(f_name) as f:
    lines = f.readlines()
    for line in lines:
        line = re.sub(r"\n", "", line)
        texts.append(line)

#TODO 第1期と第4期で word_to_id, id_to_word を共有したい
    # word_to_id, id_to_word を txt 形式で保存する？
        # 保存する関数
        # 読み込む関数
corpora, word_to_id, id_to_word = preprocess(texts)
vocab_size = len(word_to_id)
C = create_co_matrix(corpora, vocab_size)
W = sppmi(C)
wordvec_size = 100

try:
    from sklearn.utils.extmath import randomized_svd
    U, S, V = randomized_svd(W, n_components=wordvec_size, n_iter=5, random_state=None)
except:
    U, S, V = np.linalg.svd(W)

#TODO saveする名前/ディレクトリを第1期と第4期で区別
np.save("model/svd_phase1_U",U) 
np.save("model/svd_phase1_S",S) 
np.save("model/svd_phase1_V",V) 

# U[全ての単語, wordvec_size]となっている
    # U: target words, S: U, V の重要度（特異値）, V: context words
    # U*S = [全ての単語, 1次元]になる
#word_vecs = U[:, :wordvec_size] # 必要ない。Uと同じ
#word_vecs_svd = np.dot(U[:, :wordvec_size],np.sqrt(S[:wordvec_size, :wordvec_size]))
word_vecs_svd = np.dot(U,np.sqrt(S))

np.save("model/svd_phase1_wordvec",word_vecs_svd)
# 正しく学習できているか、確認
most_similar("為る", word_to_id, id_to_word, word_vecs_svd, top=5)


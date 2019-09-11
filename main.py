import sys
import numpy as np
from util import preprocess, create_co_matrix, sppmi, most_similar

f_name = sys.argv
f_name = f_name[1]
texts = []
with open(f_name) as f:
    lines = f.readlines()
    for line in lines:
        texts.append(line)

#TODO 第1期と第4期で word_to_id, id_to_word を共有したい
    # 一度に2つを学習？
        # util.py の preprocess を編集して、argument に word_to_id, id_to_word を追加？
        #TODO まずこのまま用いて、1つが正しく学習できていることを確認する
        #TODO sys.argv で2つのファイルを読み込む
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
    U, S, V = randomized_svd(W, n_vomponents=wordvec_size, n_iter=5, random_state=None)
except:
    U, S, V = np.linalg.svd(W)

#TODO saveする名前/ディレクトリを第1期と第4期で区別
np.save("U",U) 
np.save("S",S) 
np.save("V",V) 

# U[全ての単語, wordvec_size]とする（重要な部分のみ取り出し）
    # U: target words, S: U, V の重要度（特異値）, V: context words
#word_vecs = U[:, :wordvec_size]
word_vecs_svd = np.dot(U[:, :wordvec_size],np.sqrt(S[:wordvec_size, :wordvec_size]))

np.save("wordvec",word_vecs_svd)
# 正しく学習できているか、確認
most_similar("為る", word_to_id, id_to_word, word_vecs_svd, top=5)


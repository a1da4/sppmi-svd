import sys
import numpy as np
from util import preprocess, create_co_matrix, sppmi

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

#TODO 学習したデータを保存したい
    # np.save で良い？
try:
    from sklearn.utils.extmath import randomized_svd
    U, S, V = randomized_svd(W, n_vomponents=wordvec_size, n_iter=5, random_state=None)
except:
    U, S, V = np.linalg.svd(W)

# U[全ての単語, wordvec_size]とする（重要な部分のみ取り出し）
    # U: target words, S: U, V の重要度（特異値）, V: context words
    # U がそのまま単語ベクトルとなる
    #TODO U・sqrt(S)をすべき？
word_vecs = U[:, :wordvec_size]


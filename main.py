import sys
import numpy as np
import re
from util import preprocess, create_co_matrix, sppmi, most_similar

# Setting
    # window_size: 共起行列C を作成する際に考慮する窓幅
    # w2v_sg_negative_sampling: sppmi-svd は 負例k個の w2v skip-gram negative-sampling を近似している
        # 近似する負例の数 k がそのまま sppmi 計算式の MAX(0, pmi-log(k)) にあたる
    # wordvec_size: 作成する単語ベクトルのサイズで、WV = [vocab_size, wordvec_size]
window_size=10
w2v_sg_negative_sampling = 10
wordvec_size = 100

f_name = sys.argv
f_name = f_name[1]
texts = []
with open(f_name) as f:
    lines = f.readlines()
    for line in lines:
        line = re.sub(r"\n", "", line)
        texts.append(line)

corpora, word_to_id, id_to_word = preprocess(texts)
vocab_size = len(word_to_id)
if -1 in id_to_word:
    vocab_size-=1

C = create_co_matrix(corpora, vocab_size, window_size)
W = sppmi(C, w2v_sg_negative_sampling)

c_name = "model/svd_C_" + f_name.split("/")[-1][:-5]
w_name = "model/svd_W_" + f_name.split("/")[-1][:-5]
np.save(c_name, C)
np.save(w_name, W)

U, S, V = np.linalg.svd(W)

u_name = "model/svd_U_" + f_name.split("/")[-1][:-5]
s_name = "model/svd_S_" + f_name.split("/")[-1][:-5]
v_name = "model/svd_V_" + f_name.split("/")[-1][:-5]
np.save(u_name, U) 
np.save(s_name, S) 
np.save(v_name, V) 

# U[全ての単語, wordvec_size]となっている
    # U: target words, S: U, V の重要度（特異値）, V: context words
word_vecs_svd = np.dot(U,np.sqrt(np.diag(S)))

wv_name = "model/svd_WV_" + f_name.split("/")[-1][:-5]
np.save(wv_name, word_vecs_svd[:,:wordvec_size])

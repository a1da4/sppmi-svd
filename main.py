import sys
import numpy as np
import re
from util import preprocess, create_co_matrix, sppmi, most_similar

def main(f_name, threshold=False, smoothing=False, window_size=10, w2v_sgns=10, wv_size=100):
    """ create wordvec
    
    :param f_name: data path
    :param window_size: window size
    :param w2v_sgns: num of samples in w2v skip-gram negative-sampling(sgns) 
    :param wv_size: the size of wordvec WV = [vocab_size, wv_size]
    """

    texts = []
    with open(f_name) as f:
        lines = f.readlines()
        for line in lines:
            line = re.sub(r"\n", "", line)
            texts.append(line)

    corpus, word_to_id, id_to_word = preprocess(texts)
    vocab_size = len(word_to_id)
    if -1 in id_to_word:
        vocab_size-=1

    C = create_co_matrix(corpus, vocab_size, window_size)

    # threshold by min_count
    if threshold:
        C = truncate(C, threshold=threshold)

    # use smoothing or not in computing sppmi
    W = sppmi(C, w2v_sgns, smoothing=smoothing)

    c_name = "model/svd_C_" + f_name.split("/")[-1][:-5]
    w_name = "model/svd_SPPMI_" + f_name.split("/")[-1][:-5]
    np.save(c_name, C)
    np.save(w_name, W)

    try:
        from scipy.sparse.linalg import svds
        U, S, V = svds(W, k=wv_size)
    except:
        U, S, V = np.linalg.svd(W)

    u_name = "model/svd_U_" + f_name.split("/")[-1][:-5]
    s_name = "model/svd_S_" + f_name.split("/")[-1][:-5]
    v_name = "model/svd_V_" + f_name.split("/")[-1][:-5]
    np.save(u_name, U) 
    np.save(s_name, S) 
    np.save(v_name, V) 

    # U[全ての単語, wv_size]となっている
        # U: target words, S: U, V の重要度（特異値）, V: context words
    word_vecs_svd = np.dot(U,np.sqrt(np.diag(S)))

    wv_name = "model/svd_WV_" + f_name.split("/")[-1][:-5]
    np.save(wv_name, word_vecs_svd[:,:wv_size])

    return


if __name__ == '__main__':
    f_name = sys.argv[1]
    main(f_name)
     

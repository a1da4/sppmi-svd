import os
import re
import argparse
import numpy as np
from util import preprocess, create_co_matrix, truncate, sppmi, most_similar

def main(args):
    """ create wordvec
    
    :param f_name: data path
    :param window_size: window size
    :param w2v_sgns: num of samples in w2v skip-gram negative-sampling(sgns) 
    :param wv_size: the size of wordvec WV = [vocab_size, wv_size]
    """

    texts = []
    with open(args.file_path) as f:
        lines = f.readlines()
        for line in lines:
            line = re.sub(r"\n", "", line)
            texts.append(line)

    corpus, word_to_id, id_to_word = preprocess(texts)
    vocab_size = len(word_to_id)
    if -1 in id_to_word:
        vocab_size-=1

    C = create_co_matrix(corpus, vocab_size, args.window_size)

    # threshold by min_count
    if args.threshold:
        C = truncate(C, threshold=args.threshold)

    # use smoothing or not in computing sppmi
    W = sppmi(C, args.w2v_sgns, smoothing=args.smoothing)

    os.makedirs('model', exist_ok=True)
    c_name = "model/C"
    w_name = "model/SPPMI"
    np.save(c_name, C)
    np.save(w_name, W)

    try:
        from scipy.sparse.linalg import svds
        U, S, V = svds(W, k=args.wv_size)
    except:
        U, S, V = np.linalg.svd(W)

    u_name = "model/svd_U" 
    s_name = "model/svd_S"
    v_name = "model/svd_V"
    np.save(u_name, U) 
    np.save(s_name, S) 
    np.save(v_name, V) 

    word_vecs_svd = np.dot(U,np.sqrt(np.diag(S)))

    wv_name = "model/WV"
    np.save(wv_name, word_vecs_svd[:,:args.wv_size])

    return


def cli_main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--file_path', help='a path of corpus')
    parser.add_argument('--threshold', default=False, help='adopt threshold to co-occur matrix or not')
    parser.add_argument('--smoothing', default=False, help='adopt absolute discounting or not')
    parser.add_argument('--window_size', default=10, help='window size for co-occur matrix')
    parser.add_argument('--w2v_sgns', default=10, help='num of negative samples in computing SPPMI')
    parser.add_argument('--wv_size', default=100, help='size of word vector')

    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    cli_main()
     

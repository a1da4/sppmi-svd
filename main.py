import os
import re
import argparse

import numpy as np

from util import load_pickle, create_co_matrix, threshold_cooccur, sppmi, most_similar

def main(args):
    """ create word vector
    :param file_path: path of corpus
    :param window_size: window size
    :param shift: num of samples in w2v skip-gram negative-sampling(sgns) 
    :param dim: the size of wordvec WV = [vocab_size, dim]
    """
    print(args)

    print('Loading dictionary...')
    id_to_word, word_to_id = load_pickle(args.pickle_id2word)
    vocab_size = len(id_to_word)
    print(f'  {vocab_size} words')

    print('Creating co-occur matrix...')
    C = create_co_matrix(args.file_path, word_to_id, vocab_size, args.window_size)
    os.makedirs('model', exist_ok=True)
    c_name = "model/C_w-{}".format(args.window_size)
    np.save(c_name, C)

    # threshold by min_count
    if args.threshold:
        C = threshold_cooccur(C, threshold=args.threshold)

    print('Computing sppmi matrix...')
    # use smoothing or not in computing sppmi
    W = sppmi(C, args.shift, has_abs_dis=args.has_abs_dis)
    w_name = "model/SPPMI_w-{}_s-{}".format(args.window_size, args.shift)
    np.save(w_name, W)

    try:
        from scipy.sparse.linalg import svds
        U, S, V = svds(W, k=args.dim)
    except:
        U, S, V = np.linalg.svd(W)

    word_vec = np.dot(U,np.sqrt(np.diag(S)))
    wv_name = "model/WV_d-{}_w-{}_s-{}".format(args.dim, args.window_size, args.shift)
    np.save(wv_name, word_vec[:, :args.dim])

    return


def cli_main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-f', '--file_path', help='a path of corpus')
    parser.add_argument('-p', '--pickle_id2word', help='a path of index to word dictionary, dic_id2word.pkl')
    parser.add_argument('-t', '--threshold', type=int, default=0, help='adopt threshold to co-occur matrix or not')
    parser.add_argument('-a',  '--has_abs_dis', action='store_true', help='adopt absolute discounting or not')
    parser.add_argument('-w', '--window_size', type=int, default=10, help='window size for co-occur matrix')
    parser.add_argument('-s', '--shift', type=int, default=10, help='num of negative samples in computing SPPMI')
    parser.add_argument('-d', '--dim', type=int, default=100, help='size of word vector')

    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    cli_main()
     

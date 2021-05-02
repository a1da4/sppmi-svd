import argparse
import logging
import os
import re

import numpy as np
from scipy.sparse import coo_matrix

from util import create_co_matrix, load_pickle, load_matrix, most_similar, sppmi, threshold_cooccur


def main(args):
    """create word vector
    :param file_path: path of corpus
    :param window_size: window size
    :param shift: num of samples in w2v skip-gram negative-sampling(sgns)
    :param dim: the size of wordvec WV = [vocab_size, dim]
    """
    logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
    logging.info(f"[INFO] args: {args}")

    logging.info("[INFO] Loading dictionary...")
    id_to_word, word_to_id = load_pickle(args.pickle_id2word)
    vocab_size = len(id_to_word)
    logging.debug(f"[DEBUG] vocab: {vocab_size} words")

    if args.cooccur_pretrained is not None:
        logging.info("[INFO] Loading pre-trained co-occur matrix...")
        C = load_matrix(args.cooccur_pretrained, len(id_to_word))
    else:
        logging.info("[INFO] Creating co-occur matrix...")
        C = create_co_matrix(args.file_path, word_to_id, vocab_size, args.window_size)

        # threshold by min_count
        if args.threshold:
            C = threshold_cooccur(C, threshold=args.threshold)

        os.makedirs("model", exist_ok=True)
        c_name = "model/C_w-{}".format(args.window_size)
        with open(c_name, "w") as wp:
            for id, cooccur_each in enumerate(C):
                cooccur_nonzero = [
                    f"{id}:{c}" for id, c in enumerate(cooccur_each) if c > 0
                ]
                wp.write(f"{id}\t{' '.join(cooccur_nonzero)}\n")

    if args.sppmi_pretrained is not None:
        logging.info("[INFO] Loading pre-trained sppmi matrix...")
        M = load_matrix(args.sppmi_pretrained, len(id_to_word))
    else:
        logging.info("[INFO] Computing sppmi matrix...")
        # use smoothing or not in computing sppmi
        M = sppmi(C, args.shift, has_abs_dis=args.has_abs_dis, has_cds=args.has_cds)
        m_name = "model/SPPMI_w-{}_s-{}".format(args.window_size, args.shift)
        with open(m_name, "w") as wp:
            for id, sppmi_each in enumerate(M):
                sppmi_nonzero = [f"{id}:{m}" for id, m in enumerate(sppmi_each) if m > 0]
                wp.write(f"{id}\t{' '.join(sppmi_nonzero)}\n")

    logging.info("[INFO] Calculating word vector...")
    try:
        from scipy.sparse.linalg import svds

        U, S, V = svds(coo_matrix(M), k=args.dim)
    except:
        U, S, V = np.linalg.svd(coo_matrix(M))

    word_vec = np.dot(U, np.sqrt(np.diag(S)))
    wv_name = "model/WV_d-{}_w-{}_s-{}".format(args.dim, args.window_size, args.shift)
    np.save(wv_name, word_vec[:, : args.dim])

    return


def cli_main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--file_path", help="a path of corpus")
    parser.add_argument(
        "-p",
        "--pickle_id2word",
        help="a path of index to word dictionary, dic_id2word.pkl",
    )
    parser.add_argument(
        "--cooccur_pretrained", help="pre-trained cooccur matrix (file)"
    )
    parser.add_argument("--sppmi_pretrained", help="pre-trained sppmi matrix (file)")
    parser.add_argument(
        "-t",
        "--threshold",
        type=int,
        default=0,
        help="adopt threshold to co-occur matrix or not",
    )
    parser.add_argument(
        "-a",
        "--has_abs_dis",
        action="store_true",
        help="adopt absolute discounting or not",
    )
    parser.add_argument(
        "-c",
        "--has_cds",
        action="store_true",
        help="adopt contextual distributional smoothing or not",
    )
    parser.add_argument(
        "-w",
        "--window_size",
        type=int,
        default=10,
        help="window size for co-occur matrix",
    )
    parser.add_argument(
        "-s",
        "--shift",
        type=int,
        default=10,
        help="num of negative samples in computing SPPMI",
    )
    parser.add_argument(
        "-d", "--dim", type=int, default=100, help="size of word vector"
    )

    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()

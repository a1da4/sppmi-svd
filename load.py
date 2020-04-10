import re
import argparse
import numpy as np
from util import most_similar

def main(args):
    """ Display most similar k words
    """
    print(args)

    word_vec_svd = np.load(args.model_path)

    with open(args.id2word_path) as f:
        pairs = f.readlines()
        word_to_id = {}
        id_to_word = {}
        for p in pairs:
            p = re.sub(r"\n", "", p)
            p = p.split("\t")
            id = int(p[0])
            word = p[1]
            id_to_word[id] = word
            word_to_id[word] = id

    for word in word_to_id:
        most_similar(word, word_to_id, id_to_word, word_vec_svd)


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='path of wordvec model')
    parser.add_argument('--id2word_path', help='path of id2word list')
    args = parser.parse_args()
    
    main(args)


if __name__ == '__main__':
    cli_main()

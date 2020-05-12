import argparse
import re
import _pickle

from tqdm import tqdm
from collections import Counter


def main(args):
    word_freq = Counter()

    with open(args.file_path) as fp:
        for line in fp:
            line = re.sub('\n', '', line)
            words = line.split(' ')
            for word in words:
                word_freq[word] += 1

    print(f'Number of all words in the corpus: {len(word_freq)}')
    tgt_words = [w for w in word_freq if word_freq[w] >= args.threshold]
    print(f'Number of words appear more than {args.threshold}-times: {len(tgt_words)}')
    
    print('Create id2word dictionary...')
    id2word = {}
    for i in tqdm(range(len(tgt_words))):
        id2word[i] = tgt_words[i]
    
    fp = open('dic_id2word.pkl', 'wb')
    _pickle.dump(id2word, fp)


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_path', help='the path of corpus')
    parser.add_argument('-t', '--threshold', type=int, default=100, help='threshold in frequcy. the word appears more than threshold, it will be added in the id2word_dictionary.')
    
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    cli_main()

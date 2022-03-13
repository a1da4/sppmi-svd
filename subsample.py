import argparse
import logging
from collections import Counter

import numpy as np


def compute_remove_prob(word, total_freq, word2freq, t=1e-3):
    raw_freq = word2freq[word]
    corpus_freq = raw_freq / total_freq
    remove_prob = 1 - np.sqrt(t / corpus_freq)
    return remove_prob


def subsample(sentence, total_freq, word2freq, t=1e-3):
    sentence_subsampled = []
    for word in sentence.split(" "):
        remove_prob = compute_remove_prob(word, total_freq, word2freq, t)
        sample_prob = np.random.rand()
        if remove_prob < 0 or sample_prob >= remove_prob:
            sentence_subsampled.append(word)
    return sentence_subsampled


def main(args):
    logging.basicConfig(level=logging.INFO)
    logging.info(f"[main] args: {args}")

    logging.info("[main] Count (raw) word frequency...")
    word2freq = Counter()
    total_freq = 0
    with open(args.file_path) as fp:
        for line in fp:
            words = line.strip().split()
            for word in words:
                word2freq[word] += 1
                total_freq += 1
    logging.info(f"[main] - most frequent words: {word2freq.most_common(5)}")
    logging.info(f"[main] - total_freq: {total_freq}")

    logging.info("[main] Subsampling...")
    words_remove_probs = []
    for word_freq in word2freq.most_common(5):
        word = word_freq[0]
        remove_prob = compute_remove_prob(word, total_freq, word2freq, args.threshold)
        words_remove_probs.append((word, remove_prob))
    logging.info(
        f"[main] - remove probability of most frequent words: {words_remove_probs}"
    )
    sentences_subsampled = []
    with open(args.file_path) as fp:
        for line in fp:
            sentence = line.strip()
            sentence_subsampled = subsample(
                sentence, total_freq, word2freq, args.threshold
            )
            sentences_subsampled.append(sentence_subsampled)

    logging.info("[main] Save processed document...")
    with open(f"corpus_subsampled_t-{args.threshold}.txt", "w") as fp:
        for sentence_subsampled in sentences_subsampled:
            fp.write(f"{' '.join(sentence_subsampled)}\n")

    logging.info(f"[main] Count (subsampled) word frequency...")
    word2freq_subsampled = Counter()
    total_freq_subsampled = 0
    for sentence_subsampled in sentences_subsampled:
        for word in sentence_subsampled:
            word2freq_subsampled[word] += 1
            total_freq_subsampled += 1
    logging.info(f"[main] - most frequent words: {word2freq_subsampled.most_common(5)}")
    logging.info(f"[main] - total_freq: {total_freq_subsampled}")
    logging.info(
        f"[main] - {1 - total_freq_subsampled / total_freq}% words are removed"
    )


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file_path", help="path of corpus (segmented, each line = sentence)")
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=1e-3,
        help="param of subsample (default: 1e-3)",
    )

    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()

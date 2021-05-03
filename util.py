import math
import re

import _pickle
import numpy as np
from tqdm import tqdm
from math import log, log2


def load_pickle(pickle_id2word):
    """load dictionary
    :param id_to_word: dictionary(id->word)
    :param word_to_word: dictionary(word->id)

    :return: id2word, word2id
    """
    word_to_id = {}

    fp = open(pickle_id2word, "rb")
    id_to_word = _pickle.load(fp)
    for index in id_to_word:
        word_to_id[id_to_word[index]] = index
    return id_to_word, word_to_id


def load_matrix(file_matrix, V):
    """load matrix
    :param file_matrix: path of pre-trained matrix (output file)
    :param V: vocab size

    :return: matrix(list)
    """
    matrix = [[0 for _ in range(V)] for _ in range(V)]
    with open(file_matrix) as fp:
        for line in fp:
            target_id, context_id_values = line.strip().split("\t")
            context_id_values = context_id_values.split()
            for context_id_value in context_id_values:
                context_id, value = context_id_value.split(":")
                matrix[int(target_id)][int(context_id)] += float(value)

    return matrix


def create_co_matrix(file_path, word_to_id, vocab_size, window_size):
    """create co-occur matrix
    :param corpus: corpus(fixed into id)
    :param vocab_size: vocab size
    :param window_size: windows size for counting

    :return: cooccur matrix
    """
    V = len(word_to_id)
    C = [[0 for _ in range(V)] for _ in range(V)]

    with open(file_path) as fp:
        for sentence in fp:
            words = re.sub(r"\n", "", sentence).split()

            for idx, word in enumerate(words):
                if word in word_to_id:
                    word_id = word_to_id[word]
                else:
                    continue

                for i in range(1, window_size + 1):
                    left_idx = idx - i
                    right_idx = idx + i

                    if left_idx >= 0:
                        left_word = words[left_idx]
                        if left_word in word_to_id:
                            left_word_id = word_to_id[left_word]
                            C[word_id][left_word_id] += 1

                    if right_idx < len(words):
                        right_word = words[right_idx]
                        if right_word in word_to_id:
                            right_word_id = word_to_id[right_word]
                            C[word_id][right_word_id] += 1

    return C


def threshold_cooccur(C, threshold):
    """truncate cooccur matrix by threshold value
    c = c if c > threshold else 0

    :return: fixed cooccur matrix C
    """
    for i in range(len(C)):
        cooccur_each = C[i]
        C[i] = [c if c >= threshold else 0 for c in cooccur_each]

    return C


def absolute_discounting(C, i, j, d):
    """SMOOTHING: absolute discounting
    :param C: cooccur matrix
    :param i, j: index
    :param d: discounting value (0, 1)

    :param V: vocab. size
    :param N0: number of words count[word]==0

    :return: smoothed value
    """

    if C[i][j] > 0:
        return C[i][j] - d
    else:
        V = len(C)
        C_each_nonzero = [1 if c > 0 else 0 for c in C[i]]
        N0 = V - sum(C_each_nonzero)
        return d * (V - N0) / N0


def sppmi(C, k, eps=1e-8, has_abs_dis=False, has_cds=False):
    """compute Shifted Positive PMI (SPPMI)
    :param C: cooccur matrix
    :param k: number of negative samples in w2v sgns
    :param has_abs_dis: bool, do absolute discounting smoothing or not

    :return: SPPMI matrix
    """
    V = len(C)
    M = []
    Nc = [sum(cooccur_each) for cooccur_each in C]
    N = sum(Nc)

    if has_abs_dis:
        # compute constant value d
        size = len(C) * len(C[0])
        C_each_morethan1 = [sum([1 if c >= 1 else 0 for c in C_each]) for C_each in C]
        C_each_morethan2 = [sum([1 if c >= 2 else 0 for c in C_each]) for C_each in C]
        n1 = size - sum(C_each_morethan1)
        n2 = size - sum(C_each_morethan2)
        d = n1 / (n1 + 2 * n2)

    if has_cds:
        # Context Distributional Smoothing
        C_cds = [[c ** 0.75 for c in C[i]] for i in range(V)]
        Nc_cds = [sum(cooccur_each) for cooccur_each in C]
        N_cds = sum(Nc_cds)
    else:
        Nc_cds = Nc
        N_cds = N

    for i in tqdm(range(V)):
        M_each = []
        for j in range(V):
            Cwc = absolute_discounting(C, i, j, d) if has_abs_dis else C[i][j]
            shifted_pmi = log2(Cwc * N_cds / (Nc[i] * Nc_cds[j] + eps))
            shifted_positive_pmi = max(0, shifted_pmi - log(k))
            M_each.append(shifted_positive_pmi)
        M.append(M_each)

    return M


def cos_similarity(x, y, eps=1e-8):
    """compute cos similarity
    :param x, y: vector
    :param eps: tiny value to avoid deviding 0

    :return: cos similarity
    """

    nx = x / (np.sqrt(np.sum(x ** 2)) + eps)
    ny = y / (np.sqrt(np.sum(y ** 2)) + eps)
    return np.dot(nx, ny)


def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    """search most similar top-k words
    :param query: query word
    :param word_to_id: dictionary(word->id)
    :param id_to_word: dictionary(id->word)
    :param word_matrix: wordvec
    :param top: top-k

    :return: top-k words sorted cos-similarity
    """

    if query not in word_to_id:
        print("%s is not found" % query)
        return

    print("\n[query] " + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    vocab_size = len(id_to_word)
    if -1 in id_to_word:
        vocab_size -= 1

    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    count = 0
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue
        print(" %s: %.3f" % (id_to_word[i], similarity[i]))

        count += 1
        if count >= top:
            return

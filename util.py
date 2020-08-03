import numpy as np
import math
import re
import _pickle

from tqdm import tqdm

def load_pickle(pickle_id2word):
    """ load dictionary
    :param id_to_word: dictionary(id->word)
    :param word_to_word: dictionary(word->id)
    
    :return: id2word, word2id
    """ 
    word_to_id = {}

    fp = open(pickle_id2word, 'rb')
    id_to_word = _pickle.load(fp)
    for index in id_to_word:
        word_to_id[id_to_word[index]] = index
    return id_to_word, word_to_id


def create_co_matrix(file_path, word_to_id, vocab_size, window_size):
    """ create co-occur matrix
    :param corpus: corpus(fixed into id)
    :param vocab_size: vocab size
    :param window_size: windows size for counting

    :return: cooccur matrix
    """
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    import time
    start = time.time()

    with open(file_path) as fp:
        for sentence in fp:
            words = re.sub(r'\n', '', sentence).split()

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
                            co_matrix[word_id, left_word_id] += 1

                    if right_idx < len(words):
                        right_word = words[right_idx]
                        if right_word in word_to_id:
                            right_word_id = word_to_id[right_word]
                            co_matrix[word_id, right_word_id] += 1

    end = time.time()
    print(f'  {end-start} sec')
    return co_matrix


def threshold_cooccur(C, threshold):
    """ truncate cooccur matrix by threshold value
    c = c if c > threshold else 0

    :return: fixed cooccur matrix C
    """
    C = np.where(C > threshold, C, 0)

    return C


def absolute_discounting(C, i, j, d):
    """ SMOOTHING: absolute discounting
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
        V = C.shape[1]
        N0 = V - np.count_nonzero(C[i])
        return d * (V - N0) / N0



def sppmi(C, k, eps=1e-8, has_abs_dis=False, has_cds=False):
    """ compute Shifted Positive PMI (SPPMI)
    :param C: cooccur matrix
    :param k: number of negative samples in w2v sgns
    :param has_abs_dis: bool, do absolute discounting smoothing or not

    :return: SPPMI matrix
    """
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)
    Nc = np.sum(C, axis=0)
    
    if has_abs_dis:
        # compute constant value d 
        size = C.shape[0] * C.shape[1]
        n1 = size - np.count_nonzero(C-1)
        n2 = size - np.count_nonzero(C-2)
        d = n1 / (n1 + 2 * n2)
        print(f'discount value d: {d}')

    if has_cds:
        # Context Distributional Smoothing
        N = N**0.75
        Nc = Nc**0.75

    for i in tqdm(range(C.shape[0])):
        for j in range(C.shape[1]):
            Cwc = absolute_discounting(C, i, j, d) if has_abs_dis else C[i, j]
            shifted_positive_pmi = np.log2(Cwc * N / (Nc[j]*Nc[i]) + eps)
            M[i, j] = max(0, shifted_positive_pmi - math.log(k))

    return M


def cos_similarity(x, y, eps=1e-8):
    """ compute cos similarity
    :param x, y: vector
    :param eps: tiny value to avoid deviding 0

    :return: cos similarity
    """
    
    nx = x / (np.sqrt(np.sum(x ** 2)) + eps)
    ny = y / (np.sqrt(np.sum(y ** 2)) + eps)
    return np.dot(nx, ny)


def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    """ search most similar top-k words
    :param query: query word
    :param word_to_id: dictionary(word->id)
    :param id_to_word: dictionary(id->word)
    :param word_matrix: wordvec
    :param top: top-k

    :return: top-k words sorted cos-similarity
    """
    
    if query not in word_to_id:
        print('%s is not found' % query)
        return

    print('\n[query] ' + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    vocab_size = len(id_to_word)
    if -1 in id_to_word:
        vocab_size-=1

    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    count = 0
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue
        print(' %s: %.3f' % (id_to_word[i], similarity[i]))

        count += 1
        if count >= top:
            return

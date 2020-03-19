import numpy as np
import math
from tqdm import tqdm

def preprocess(texts):
    """ fix text -> id

    :param texts: sentences
    :param word_to_id: dictionary(word->id)
    :param id_to_word: dictionary(id->word)
    
    :return: corpus(fixed into id), word2id, id2word
    """ 
    make_new_dic = 0
    try:
        with open("id_to_word.txt") as f:
            pairs = f.readlines()
            word_to_id = {}
            id_to_word = {}
            import re
            for p in pairs:
                p = re.sub(r"\n", "", p)
                p = p.split("\t")
                id = int(p[0])
                word = p[1]
                id_to_word[id] = word
                word_to_id[word] = id

    except:
        word_to_id = {}
        id_to_word = {}
        make_new_dic = 1

    #corpora = None
    corpora = []
    
    # out of vocab
    word_to_id['#'] = -1
    id_to_word[-1] = '#'

    for text in texts:
        words = text.split(" ")
        words = [w for w in words if len(w) > 0]
        for word in words:
            if word not in word_to_id:
                if make_new_dic:
                    new_id = len(word_to_id)-1
                    word_to_id[word] = new_id
                    id_to_word[new_id] = word
                else:
                    words[words.index(word)] = '#'
        corpora.append(np.array([word_to_id[w] for w in words]))
    
    if make_new_dic:
        with open("id_to_word.txt", "w") as f:
            for id in id_to_word:
                f.write(f"{id}\t{id_to_word[id]}")
                f.write("\n")

    return corpora, word_to_id, id_to_word


def create_co_matrix(corpora, vocab_size, window_size):
    """create cooccur matrix

    :param corpus: corpus(fixed into id)
    :param vocab_size: vocab size
    :param window_size: windows size for counting

    :return: cooccur matrix
    """
    corpus_size = sum([len(c) for c in corpora])
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for corpus in corpora:
        for idx, word_id in tqem(enumerate(corpus)):
            if word_id == -1:
                continue
            for i in range(1, window_size + 1):
                left_idx = idx - i
                right_idx = idx + i

                if left_idx >= 0:
                    left_word_id = corpus[left_idx]
                    if left_word_id != -1:
                        co_matrix[word_id, left_word_id] += 1

                if right_idx < len(corpus):
                    right_word_id = corpus[right_idx]
                    if right_word_id != -1:
                        co_matrix[word_id, right_word_id] += 1

    return co_matrix


def truncate(C, threshold):
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

    :return: smoothed value
    """

    discounting = max(C[i][j] - d, 0)
    smoothing = d * np.count_nonzero(C[i]) / C.shape[1]

    return discounting + smoothing


def sppmi(C, k, eps=1e-8, smoothing=False):
    """ compute Shifted Positive PMI (SPPMI)

    :param C: cooccur matrix
    :param k: number of negative samples in w2v sgns

    :return: SPPMI matrix
    """
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)
    Nc = np.sum(C, axis=0)
    
    if smoothing:
        # compute constant value d 
        size = C.shape[0] * C.shape[1]
        n1 = size - np.count_nonzero(C-1)
        n2 = size - np.count_nonzero(C-2)
        d = n1 / (n1 + 2 * n2)
        print(f'discount value d: {d}')


    for i in tqdm(range(C.shape[0])):
        for j in range(C.shape[1]):
            Cwc = absolute_discounting(C, i, j, d) if smoothing else C[i, j]
            #pmi = np.log2(C[i, j] * N / (Nc[j]*Nc[i]) + eps)
            pmi = np.log2(Cwc * N / (Nc[j]*Nc[i]) + eps)
            M[i, j] = max(0, pmi - math.log(k))

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

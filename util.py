import numpy as np
import math
def preprocess(texts):
    '''生のコーパス（文毎に区切られているもの）をid化
    '''

    word_to_id = {}
    id_to_word = {}

    #corpora = None
    corpora = []
    
    for text in texts:
        words = text.split(" ")

        for word in words:
            if word not in word_to_id:
                new_id = len(word_to_id)
                word_to_id[word] = new_id
                id_to_word[new_id] = word
        #if corpora == None:
            #corpora = np.array([word_to_id[w] for w in words])
            #line = 1
        #elif line == 1:
            #corpora = 
        #else:
            #corpora = 
                corpora.append(np.array([word_to_id[w] for w in words]))
    #return corpus, word_to_id, id_to_word
    return corpora, word_to_id, id_to_word


def create_co_matrix(corpus, vocab_size, window_size=10):
    '''共起行列の作成
    :param corpus: コーパス（単語IDのリスト）
    :param vocab_size:語彙数
    :param window_size:ウィンドウサイズ（ウィンドウサイズが1のときは、単語の左右1単語がコンテキスト）
    :return: 共起行列
    '''
    #corpus_size = len(corpus)
    corpus_size = sum([len(c) for c in corpora])
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for corpus in corpora:
        for idx, word_id in enumerate(corpus):
            for i in range(1, window_size + 1):
                left_idx = idx - i
                right_idx = idx + i

                if left_idx >= 0:
                    left_word_id = corpus[left_idx]
                    co_matrix[word_id, left_word_id] += 1

                if right_idx < corpus_size:
                    right_word_id = corpus[right_idx]
                    co_matrix[word_id, right_word_id] += 1

    return co_matrix


def sppmi(C, verbose=False, k=10, eps=1e-8):
    '''SPPMI（正の相互情報量-log(負例)）の作成
    :param C: 共起行列
    :param verbose: 進行状況を出力するかどうか
    :param k: 負例の数
    :return:
    '''
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C, axis=0)
    total = C.shape[0] * C.shape[1]
    cnt = 0

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / (S[j]*S[i]) + eps)
            M[i, j] = max(0, pmi - math.log(k))

            if verbose:
                cnt += 1
                if cnt % (total//100) == 0:
                    print('%.1f%% done' % (100*cnt/total))
    return M


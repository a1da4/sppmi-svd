import numpy as np
import math
def preprocess(texts, word_to_id=None, id_to_word=None):
    '''生のコーパス（文毎に区切られているもの）をid化
    '''
    if word_to_id == None and id_to_word == None:
        word_to_id = {}
        id_to_word = {}

    #corpora = None
    corpora = []
    
    for text in texts:
        words = text.split(" ")
        words = [w for w in words if len(w) > 0]

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


def create_co_matrix(corpora, vocab_size, window_size=10):
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

                if right_idx < len(corpus):
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


def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    '''類似単語の検索
    :param query: クエリ（テキスト）
    :param word_to_id: 単語から単語IDへのディクショナリ
    :param id_to_word: 単語IDから単語へのディクショナリ
    :param word_matrix: 単語ベクトルをまとめた行列。各行に対応する単語のベクトルが格納されていることを想定する
    :param top: 上位何位まで表示するか
    '''
    if query not in word_to_id:
        print('%s is not found' % query)
        return

    print('\n[query] ' + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    vocab_size = len(id_to_word)

    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    count = 0
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue
        print(' %s: %s' % (id_to_word[i], similarity[i]))

        count += 1
        if count >= top:
            return

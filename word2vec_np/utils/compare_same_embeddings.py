"""
Compare the embeddings of two runs

"""
import numpy as np
from scipy import spatial


def get_most_common_words(dictionary, n_words):
    return dictionary[:n_words]


def get_random_words(dictionary, n_words):
    rand_indices = np.random.randint(len(dictionary), size=n_words)
    return dictionary[rand_indices]


def get_cosine_similarity(dict_1, W1_1, dict_2, W1_2):
    for ind_1, word_1 in enumerate(dict_1):
        emb_1 = W1_1[ind_1, :]
        # Find the same word in the other dictionary
        # (while they are sorted by frequency, they are not necessarily aligned)
        ind_2 = dict_2.index(word_1)
        word_2 = dict_2[ind_2]
        emb_2 = W1_2[ind_2, :]
        similarity = 1 - spatial.distance.cosine(emb_1, emb_2)
        print('word1 = %s, word2 = %s, cosine similarity = %f' % (word_1, word_2, similarity))


def main():
    # Dictionary and embeddings for run #1
    dict_file_1 = '../data-out/2018-03-08/run6/replicate/dictionary.txt'
    w1_file_1 = '../data-out/2018-03-08/run6/mine/w1.npy'

    # Dictionary and the embeddings for run #2
    dict_file_2 = '../data-out/2018-03-07/run4/dictionary.txt'
    w1_file_2 = '../data-out/2018-03-07/run4/w1.npy'

    dict_1 = [line.rstrip('\n') for line in open(dict_file_1)]
    W1_1 = np.load(w1_file_1)

    dict_2 = [line.rstrip('\n') for line in open(dict_file_2)]
    W1_2 = np.load(w1_file_2)

    # Drop the first, second and third rows of W1 cause it is a dummy for the 'PAD', 'UNK' and 'EOS' symbols
    W1_1 = np.delete(W1_1, 0, 0)
    W1_1 = np.delete(W1_1, 0, 0)
    W1_1 = np.delete(W1_1, 0, 0)

    W1_2 = np.delete(W1_2, 0, 0)
    W1_2 = np.delete(W1_2, 0, 0)
    W1_2 = np.delete(W1_2, 0, 0)

    # Drop the first element in dict_1 cause it's the 'UNK' symbol
    dict_1.remove('UNK')
    dict_1.remove('EOS')
    dict_2.remove('UNK')
    dict_2.remove('EOS')

    get_cosine_similarity(dict_1, W1_1, dict_2, W1_2)


main()

"""Helper script to retrieve nearest
neighbours to the given word in the embedding space
"""

import numpy as np
from scipy import spatial


def get_most_common_words(dictionary, n_words):
    return dictionary[:n_words]


def get_random_words(dictionary, n_words):
    rand_indices = np.random.randint(len(dictionary), size=n_words)
    return [dictionary[i] for i in rand_indices]


def get_closest_words_euler(dictionary, word, n_words, w1):
    word_vec = w1[dictionary.index(word), :]
    distances = np.linalg.norm(w1 - word_vec, axis=1)
    sorted_indices = np.argsort(distances)
    return [dictionary[i] for i in sorted_indices[:n_words]]


def get_closest_words_cosine(dictionary, word, n_words, w1):
    word_vec = w1[dictionary.index(word), :]
    # Cosine similarity is defined as cos(angle between two vecs)
    # 1 means exactly the same, -1 means exactly the opposite.
    # We append minus in from to be able to use argsort
    similarities = []
    for i in range(w1.shape[0]):
        similarities.append(-1 + spatial.distance.cosine(w1[i,:], word_vec))
    sorted_indices = np.argsort(similarities)
    return [dictionary[i] for i in sorted_indices[:n_words]]


def main():
    # Dictionary and embeddings for my code
    dict_file_mine = '../data-out/2018-03-24/new/dictionary.txt'
    w1_file_mine = '../data-out/2018-03-24/new/w1.npy'
    # Dictionary and embeddings from word2vec code
    dict_file_w2v = '../data-out/2018-03-24/old/dictionary.txt'
    w1_file_w2v = '../data-out/2018-03-24/old/w1.npy'

    # Load my dictionary and embeddings
    dictionary_mine = [line.rstrip('\n') for line in open(dict_file_mine)]
    w1_mine = np.load(w1_file_mine)
    w1_mine = np.delete(w1_mine, 0, 0)
    w1_mine = np.delete(w1_mine, 0, 0)
    w1_mine = np.delete(w1_mine, 0, 0)
    # dictionary_mine.remove('UNK')
    # dictionary_mine.remove('EOS')
    # Load word2vec dictionary and embeddings
    # dictionary_w2v = [line.rstrip('\n') for line in open(dict_file_w2v)]
    # w1_w2v = np.loadtxt(w1_file_w2v)
    dictionary_w2v = [line.rstrip('\n') for line in open(dict_file_w2v)]
    w1_w2v = np.load(w1_file_w2v)
    w1_w2v = np.delete(w1_w2v, 0, 0)
    w1_w2v = np.delete(w1_w2v, 0, 0)
    w1_w2v = np.delete(w1_w2v, 0, 0)

    n_freq_words = 50
    n_closest_words = 20

    grouped_words = []

    # Get top 50 most common words from one of the dictionaries
    words = get_most_common_words(dictionary_mine, n_freq_words)
    for word in words:
        closest_words_mine = get_closest_words_euler(dictionary_mine, word, n_closest_words, w1_mine)
        closest_words_w2v = get_closest_words_euler(dictionary_w2v, word, n_closest_words, w1_w2v)
        grouped_words.append((word, closest_words_mine, closest_words_w2v))

    for grouped_word in grouped_words:
        print(grouped_word)

    # Dump the result into a file
    eval_file = open('../data-out/2018-03-24/closest-words.txt', 'w+')
    for grouped_word in grouped_words:
        eval_file.write(grouped_word[0] + ': \n')
        eval_file.write(', '.join(grouped_word[1]) + '\n')
        eval_file.write(', '.join(grouped_word[2]) + '\n\n')
    eval_file.close()


if __name__ == "__main__":
    main()

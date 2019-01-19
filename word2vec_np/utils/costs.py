import numpy as np
from word2vec_np.utils.activations import sigmoid


def cost_ns(U):
    # Get the the batch size.
    m = U.shape[0]
    # The first column of U corresponds to true labels, the rest are for negative samples
    cost = np.sum(-np.log(sigmoid(U[:, 0])) - np.sum(np.log(sigmoid(-U[:, 1:])), axis=1), axis=0)
    return cost / m


def cost_sm(U, YT):
    # Get the the batch size.
    m = U.shape[1]
    cost = np.sum(np.log(np.sum(np.exp(U), axis=0)) - np.sum(np.multiply(YT, U), axis=0))
    return cost / m

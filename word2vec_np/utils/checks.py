"""Helper scripts to do some sanity checks, check gradients, etc."""

import word2vec_np.utils.propagate as propagate
import word2vec_np.utils.costs as costs
import numpy as np


def check_gradients(parameters, grads, X, Y, epsilon=1e-7):
    # Get the size of the vocabulary
    V = parameters["V"]
    # Get the size of the embedding
    N = parameters["N"]
    # Get the number of context words
    n_context_words = parameters['n_context_words']
    # Get the number of negative samples
    n_neg_samples = parameters['n_neg_samples']
    # Get the mini-batch size
    m = X.shape[0]
    # Get the weight matrices
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    # Make sure the shapes are right
    assert W1.shape == (V + 1, N)
    assert W2.shape == (N, V + 1)
    assert X.shape == (m, 2 * n_context_words)
    assert Y.shape == (m, n_neg_samples + 1)

    # Get the gradients obtained through back propagation
    dE_dW1 = grads["dE_dW1"]
    dE_dW2 = grads["dE_dW2"]

    # Make sure the sizes of the gradients are right
    assert dE_dW1.shape == W1.shape
    assert dE_dW2.shape == W2.shape

    # Define arrays for storing the approximated gradients
    dE_dW1_A = np.zeros(W1.shape, dtype=np.float32)
    dE_dW2_A = np.zeros(W2.shape, dtype=np.float32)

    # Note for W1 we skip the first row cause that's for the 'PAD' symbol.
    for i in range(1, W1.shape[0]):
        for j in range(W1.shape[1]):
            W1_plus = np.copy(W1)
            W1_plus[i, j] += epsilon
            parameters["W1"] = W1_plus
            output = propagate.forward_ns(X, Y, parameters)
            cost_plus = costs.cost_ns(output["U"])

            W1_minus = np.copy(W1)
            W1_minus[i, j] -= epsilon
            parameters["W1"] = W1_minus
            output = propagate.forward_ns(X, Y, parameters)
            cost_minus = costs.cost_ns(output["U"])
            dE_dW1_A[i, j] = (cost_plus - cost_minus) / (2 * epsilon)

    parameters["W1"] = W1

    for i in range(W2.shape[0]):
        for j in range(W2.shape[1]):
            W2_plus = np.copy(W2)
            W2_plus[i, j] += epsilon
            parameters["W2"] = W2_plus
            output = propagate.forward_ns(X, Y, parameters)
            cost_plus = costs.cost_ns(output["U"])

            W2_minus = np.copy(W2)
            W2_minus[i, j] -= epsilon
            parameters["W2"] = W2_minus
            output = propagate.forward_ns(X, Y, parameters)
            cost_minus = costs.cost_ns(output["U"])
            dE_dW2_A[i, j] = (cost_plus - cost_minus) / (2 * epsilon)

    numerator = np.linalg.norm(dE_dW1 - dE_dW1_A)
    denominator = np.linalg.norm(dE_dW1) + np.linalg.norm(dE_dW1_A)
    difference_W1 = numerator / denominator

    numerator = np.linalg.norm(dE_dW2 - dE_dW2_A)
    denominator = np.linalg.norm(dE_dW2) + np.linalg.norm(dE_dW2_A)
    difference_W2 = numerator / denominator

    assert difference_W1 < epsilon, "Gradient for W1 is wrong!"
    assert difference_W2 < epsilon, "Gradient for W2 is wrong!"

    print(difference_W1, difference_W2)


def check_word_counts(word_count, word_count_w2v):
    for ind, (word, count) in enumerate(word_count):
        assert word == word_count_w2v[ind][0]
        assert count == word_count_w2v[ind][1]


def check_dictionaries(dict1, dict2):
    # Assert that the dictionaries are the same size.
    # Note that the order is not necessarily the same
    assert len(dict1) == len(dict2)

from activations import softmax
from activations import sigmoid
import numpy as np


def forward_sm(X, parameters):
    """
    Forward propagation for softmax training.

    :param X: Many-Hot encoded input vector X of shape (V, m)
    :param parameters: dictionary containing V, N, m and the weight matrices
    :return output: dictionary containing the values of H, U and Y
    """
    # Get the size of the vocabulary
    V = parameters["V"]
    # Get the size of the embedding
    N = parameters["N"]
    # Get the mini-batch size
    m = parameters["m"]
    # Get the weight matrices
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    # Check that the sizes are correct
    assert (W1.shape == (V, N))
    assert (W2.shape == (N, V))

    # Propagate forward to the hidden layer
    H = np.dot(np.transpose(W1), X)
    assert (H.shape == (N, m))

    # Propagate to the output layer
    U = np.dot(np.transpose(W2), H)
    assert (U.shape == (V, m))

    # Apply softmax to the output layer.
    Y = softmax(U)
    assert (Y.shape == (V, m))

    output = {"H": H,
              "U": U,
              "Y": Y}

    return output


def forward_ns(X, Y, parameters):
    """
    Forward propagation for negative sampling.

    :param X: A matrix of shape (m, 2 * n_context_words) where each row contains the indices of context words
              (padded with zeros) for that training example.
    :param Y: A matrix of shape (m, n_neg_samples + 1) where each row contains the indices of negative samples
                 for that training example.
    :param parameters: dictionary containing bunch of parameters and the weight matrices
    :return output: dictionary containing the values of H and U
    """
    # Get the size of the vocabulary
    vocab_size = parameters['vocab_size']
    # Get the size of the embedding
    embedding_size = parameters['embedding_size']
    # Get the number of context words
    context_window = parameters['context_window']
    # Get the number of negative samples
    neg_samples = parameters['neg_samples']
    # Get the weight matrices
    W1 = parameters['W1']
    W2 = parameters['W2']
    # Get the mini-batch size
    m = X.shape[0]

    # Check that the sizes are correct
    assert W1.shape == (vocab_size + 1, embedding_size)
    assert W2.shape == (embedding_size, vocab_size + 1)
    assert X.shape == (m, 2 * context_window)
    assert Y.shape == (m, neg_samples + 1)

    # Propagate forward to the hidden layer
    # First, find the number of non-zero elements for each training example
    n_cw = np.sum(X != 0, axis=1)
    # Extend dimension along axis 1
    n_cw = np.expand_dims(n_cw, axis=1)
    assert n_cw.shape == (m, 1)

    # With this, now we can compute the hidden layer averaged over context words.
    H = np.sum(W1[X], axis=1) / n_cw
    assert H.shape == (m, embedding_size)

    # Propagate to the output layer
    # First, compute transpose of W2
    W2T = W2.T
    # Break it into m arrays sampled from Y
    WR = W2T[Y]
    assert WR.shape == (m, neg_samples + 1, embedding_size)
    # Expand H, breaking down the rows into another dim
    HR = np.expand_dims(H, axis=1)
    assert HR.shape == (m, 1, embedding_size)
    # Now we can do element-wise multiplication (this broadcasts H across n_neg_samples + 1)
    UR = np.multiply(WR, HR)
    assert UR.shape == (m, neg_samples + 1, embedding_size)
    # Finally compute the sum along the third axis (this indirectly computes the dot product for each m)
    U = np.sum(UR, axis=2)
    assert U.shape == (m, neg_samples + 1)

    output = {"H": H,
              "U": U,
              "n_cw": n_cw}

    return output


def backward_ns(X, Y, parameters, output):

    # Get the size of the vocabulary
    vocab_size = parameters['vocab_size']
    # Get the size of the embedding
    embedding_size = parameters['embedding_size']
    # Get the number of context words
    context_window = parameters['context_window']
    # Get the number of negative samples
    neg_samples = parameters['neg_samples']
    # Get the weight matrices
    W1 = parameters['W1']
    W2 = parameters['W2']
    # Get the mini-batch size
    m = X.shape[0]

    dE_dW1 = np.zeros(W1.shape, dtype=np.float32)
    dE_dW2 = np.zeros(W2.shape, dtype=np.float32)

    # Get the hidden and the output layers
    # and the number of context words per training example.
    H = output["H"]
    U = output["U"]
    n_cw = output["n_cw"]

    # Make sure the shapes are right
    assert W1.shape == (vocab_size + 1, embedding_size)
    assert W2.shape == (embedding_size, vocab_size + 1)
    assert X.shape == (m, 2 * context_window)
    assert Y.shape == (m, neg_samples + 1)
    assert H.shape == (m, embedding_size)
    assert U.shape == (m, neg_samples + 1)
    assert n_cw.shape == (m, 1)

    # First, compute the derivatives with respect to the weights of the hidden-to-output weight matrix.
    # Step 1: First compute the derivatives with respect to U
    dE_dU = sigmoid(U)
    # Step 2: Subtract 1 from the first column (these are the values for the true labels)
    dE_dU[:, 0] = dE_dU[:, 0] - 1
    # Step 3: Average over training examples in mini-batch
    dE_dU = dE_dU / m
    assert dE_dU.shape == (m, neg_samples + 1)
    # Step 3: Compute the gradients
    dE_dW2T = dE_dW2.T
    for m_ind in range(m):
        dE_dW2T[Y[m_ind, :]] += np.dot(np.transpose(dE_dU), H)
    dE_dW2 = dE_dW2T.T
    assert dE_dW2.shape == W2.shape, "Gradient matrix dE_dW2 is not of the right size"

    # Second, compute the derivatives with respect to the weights of the input-to-hidden weight matrix.
    # Step 1: Take only the relevant columns from W2 weights matrix
    W2T = W2.T
    W2TR = W2T[Y]
    assert W2TR.shape == (m, neg_samples + 1, embedding_size)
    # Step 2: Add axis to dE_dU
    dE_dU = np.expand_dims(dE_dU, axis=2)
    assert dE_dU.shape == (m, neg_samples + 1, 1)
    # Step 3: Do element-wise multiplication between W2TR and dE_dU and sum across axis 1
    dE_dH = np.sum(np.multiply(W2TR, dE_dU), axis=1)
    assert dE_dH.shape == (m, embedding_size)
    # Step 3: Compute the derivatives with respect to the input-to-hidden matrix
    # I can't figure this damn thing out without looping over training examples
    for m_ind in range(m):
        dE_dW1[X[m_ind, :]] += dE_dH[m_ind, :] / n_cw[m_ind, 0]
    # The operation above also fills the 0-th row of dE_dW1 due to numpy indexing, but we don't care about that
    # so make it zero just to keep it clean.
    dE_dW1[0, :] = 0
    assert dE_dW1.shape == W1.shape, "Gradient matrix dE_dW1 is not of the right size"

    grads = {"dE_dW1": dE_dW1,
             "dE_dW2": dE_dW2}

    return grads


def backward_sm(parameters, output, X, YT):

    # Get the size of the vocabulary
    V = parameters["V"]
    # Get the size of the embedding
    N = parameters["N"]
    # Get the mini-batch size
    m = parameters["m"]
    # Get the weight matrices
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    # Get the output and the hidden layers
    H = output["H"]
    Y = output["Y"]
    U = output["U"]

    # Make sure the shapes are right
    assert W1.shape == (V, N)
    assert W2.shape == (N, V)
    assert X.shape == (V, m)
    assert H.shape == (N, m)
    assert Y.shape == (V, m)
    assert YT.shape == (V, m)

    dE_dW2 = (1 / m) * np.dot(H, np.transpose(Y - YT))
    dE_dW1 = (1 / m) * np.dot(X, np.transpose(np.dot(W2, (Y - YT))))

    # Make sure the computed matrices of derivatives are of same shapes as the originals
    assert dE_dW1.shape == W1.shape, "Gradient matrix dE_dW1 is not of the right size"
    assert dE_dW2.shape == W2.shape, "Gradient matrix dE_dW2 is not of the right size"

    grads = {"dE_dW1": dE_dW1,
             "dE_dW2": dE_dW2}

    return grads


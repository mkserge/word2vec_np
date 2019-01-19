import word2vec_np.utils.data as data
import word2vec_np.utils.propagate as propagate
import word2vec_np.utils.costs as costs
import word2vec_np.utils.weights as weights
import argparse
import sys
import time
import logging
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--train_file', type=str, help='The training file containing the data.')
parser.add_argument('--log_file', type=str, help='The log file.')
parser.add_argument('--dict_file', type=str, help='The file to store the dictionary.')
parser.add_argument('--w1_file', type=str, help='The file to store the final w1 matrix.')
parser.add_argument('--w2_file', type=str, help='The file to store the final w2 matrix.')
parser.add_argument('--x_file', type=str, help='The file to store the x matrix for training.')
parser.add_argument('--y_file', type=str, help='The file to store the y matrix for training.')
parser.add_argument('--yneg_file', type=str, help='The file to store the yneg matrix for training.')
parser.add_argument('--vocab_size', type=int, default=0, help='The vocabulary size.')
parser.add_argument('--emb_size', type=int, default=300, help='The size of the embeddings.')
parser.add_argument('--epochs', type=int, default=5, help='The number of epochs to train on.')
parser.add_argument('--neg_samples', type=int, default=5, help='The number of negative samples to use.')
parser.add_argument('--context_window', type=int, default=20, help='The context window size (on each side).')
parser.add_argument('--batch_size', type=int, default=1, help='number of training samples in a batch.')
parser.add_argument('--ns_param', type=float, default=0.75, help='The negative sampling parameter.')
parser.add_argument('--ds_param', type=float, default=0.001, help='The down-sampling parameter.')
parser.add_argument('--start_learning_rate', type=float, default=0.05, help='The starting learning rate.')
parser.add_argument('--decay_rate', type=float, default=0.5,
                    help='The decay rate, if doing our owm adaptive learning.')
parser.add_argument('--use_w2v_weights', action='store_true', help='Use word2vec generated weights and dictionary.')
parser.add_argument('--load_data', action='store_true', help='Load pre-generated data.')
parser.add_argument('--w2v_dict_file', type=str, help='File where the word2vec dictionary is stored.')
parser.add_argument('--w2v_w1_file', type=str, help='File where the word2vec weights are stored.')


def setup_logger(args):
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)

    # Create handlers. We output to the file everything up to DEBUG.
    file_handler = logging.FileHandler(args.log_file)
    file_handler.setLevel(logging.DEBUG)
    # Stdout gets everything up to INFO.
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)

    # Create logging formats
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stdout_formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    stdout_handler.setFormatter(stdout_formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    return logger


def main():
    args = parser.parse_args()

    np.random.seed(1)
    logger = setup_logger(args)
    logger.info('Processing data.')
    start_time = time.time()

    # Get a list of the sentences from the corpus
    sentences = data.get_sentences_from_file(args.train_file)
    # Get the list of the words from the corpus
    words = data.get_words_from_sentences(sentences)
    # Get the total number of words in the corpus
    n_total_words = len(words)
    # Get the dictionary and the word counts
    dictionary, reversed_dictionary, word_count = data.get_dictionaries(words, args.vocab_size)
    # Get the vocabulary size
    vocab_size = len(word_count)
    # Get the training data
    x_train, y_train, y_neg = data.get_data(sentences, n_total_words,
                                            {'dictionary': dictionary, 'word_count': word_count}, args)
    # Get the number of training samples
    n_training_samples = len(x_train)
    # Split training data into mini-batches
    mini_batches = data.get_mini_batches(x_train, y_train, y_neg, args.batch_size, shuffled=False)

    elapsed_time = time.time() - start_time
    logger.info('Data processed in %d seconds' % elapsed_time)

    # Log the parameters for the run.
    logger.debug('Train file                   : %s' % args.train_file)
    logger.debug('Log file                     : %s' % args.log_file)
    logger.debug('W1 file                      : %a' % args.w1_file)
    logger.debug('W2 file                      : %a' % args.w2_file)
    logger.debug('Vocabulary size              : %d' % vocab_size)
    logger.debug('Total number of words        : %d' % n_total_words)
    logger.debug('Embedding size               : %d' % args.embedding_size)
    logger.debug('Number of epochs             : %d' % args.epochs)
    logger.debug('Number of mini-batches       : %d' % len(mini_batches))
    logger.debug('Number of training examples  : %d' % n_training_samples)
    logger.debug('Context words                : %d' % args.context_window)
    logger.debug('Negative samples             : %d' % args.neg_samples)
    logger.debug('Initial learning rate        : %f' % args.start_learning_rate)
    logger.debug('Decay rate                   : %f' % args.decay_rate)
    logger.debug('Negative sampling param      : %f' % args.ns_param)
    logger.debug('Down sampling param          : %f' % args.ds_param)

    # Initialize the weights of the model
    logger.info('Initializing weight matrices.')
    W1, W2 = weights.init(vocab_size, args)
    assert W1.shape == (vocab_size + 1, args.embedding_size)
    assert W2.shape == (args.embedding_size, vocab_size + 1)
    logger.info('Weight matrices initialized.')

    parameters = {'W1': W1,
                  'W2': W2,
                  'vocab_size': vocab_size,
                  'embedding_size': args.embedding_size,
                  'context_window': args.context_window,
                  'neg_samples': args.neg_samples}

    start_time = time.time()
    for n_epoch in range(0, args.epochs):
        epoch_cost = 0
        learning_rate = 0
        for ind, (X, YT, YNEG) in enumerate(mini_batches):
            # Adapt learning rate.
            if args.batch_size == 1:
                # If batch-size is one, we are dealing with stochastic gradient
                # descent and so we do adaptive learning rate exactly as in word2vec.
                learning_rate = args.start_learning_rate * \
                                (1 - (n_epoch * n_training_samples + ind) / float(args.epochs * n_training_samples + 1))
            else:
                # Otherwise, we do adaptive learning with our own decay as a hyper-parameter
                learning_rate = args.start_learning_rate / (1 + args.decay_rate * n_epoch)

            if learning_rate < args.start_learning_rate * 0.0001:
                learning_rate = args.start_learning_rate * 0.0001

            if args.neg_samples > 0:
                # Concatenate true labels as the first column of YNEG
                Y = np.concatenate((YT, YNEG), axis=1)
                output = propagate.forward_ns(X, Y, parameters)
                U = output["U"]
                cost = costs.cost_ns(U)
                grads = propagate.backward_ns(X, Y, parameters, output)
                # check_gradients(parameters, grads, X, Y)
                parameters = weights.update(parameters, grads, learning_rate)
            else:
                output = propagate.forward_sm(X, parameters)
                U = output["U"]
                Y = output["Y"]
                cost = costs.cost_sm(U, YT)
                grads = propagate.backward_sm(parameters, output, X, YT)
                parameters = weights.update(parameters, grads, learning_rate)
            epoch_cost += cost    
        # Log the average cost per epoch, every epoch
        logger.info('At epoch %d cost is %f, learning rate is %f' %
                    (n_epoch, epoch_cost / n_training_samples, learning_rate))

    elapsed_time = time.time() - start_time
    logger.info('Optimization time for %d epochs was %d seconds' % (args.epochs, elapsed_time))

    weights.save(parameters, args)


if __name__ == "__main__":
    main()

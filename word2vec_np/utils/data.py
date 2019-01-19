import word2vec_np.utils.checks as checks
import collections
import numpy as np
import math
import time
import logging


def get_sentences_from_file(train_file):
    """ Returns a list of sentences from an input file.

    Args:
        train_file:  A path to a file
    Returns:
        A list of sentences as they appear in the input.
    """

    # Read the sentences from the input file (assumed to be a sentence per line)
    sentences = [line.rstrip('\n') for line in open(train_file)]
    return sentences


def get_words_from_file(train_file):
    """ Returns a list of words from input sentences.

    Args:
        train_file:  A path to a file
    Returns:
        A list of words as they appear in the input.
    """
    words = []
    sentences = get_sentences_from_file(train_file)
    for sentence in sentences:
        sentence_words = sentence.split()
        words.extend(sentence_words)
    return words


def get_words_from_sentences(sentences):
    """ Returns a list of words from a list of sentences.

    Args:
        sentences:  A list of sentences
    Returns:
        A list of words as they appear in the input.
    """
    words = []
    for sentence in sentences:
        sentence_words = sentence.split()
        words.extend(sentence_words)
    return words


def get_indexed_sentences(sentences, dictionaries, downsample=True):
    logger = logging.getLogger('main')
    logger.info('Indexing input sentences...')
    start_time = time.time()
    num_words = 0
    indexed_sentences = []
    dictionary = dictionaries['dictionary']
    keep_probs = dictionaries['keep_probs']
    for sentence in sentences:
        indexed_sentence = []
        sentence_words = sentence.split()
        for word in sentence_words:
            word_ind = dictionary.get(word, 1)
            # 'UNK' tokens are always removed as we don't train on them
            if word_ind == 1:
                continue
            if downsample:
                random_number = np.random.rand()
                if keep_probs[word_ind - 2] < random_number:
                    continue
            indexed_sentence.append(word_ind)
        # Sentences consisting of a single word (or no words)
        # are ignored since we cannot build training examples from them.
        if len(indexed_sentence) > 1:
            indexed_sentences.append(indexed_sentence)
            num_words += len(indexed_sentence)
    elapsed_time = time.time() - start_time
    logger.info('Finished indexing input sentences in %d seconds' % elapsed_time)
    return indexed_sentences, num_words


def save_word_counts(word_count, dict_file):
    """ Saves the dictionary into a file.
    The word_count and dictionary have the same ordering
    except that dictionary has extra 'PAD' symbol at index 0


    Args:
        word_count: List of (word, count) tuples
        dict_file:  Path to the output file.
    """
    dict_file = open(dict_file, 'w+')
    for word, count in word_count:
        dict_file.write(word + ' ' + str(count) + '\n')
    dict_file.close()


def save_dictionary(word_count, dict_file):
    """Saves the dictionary into a file.
    The word_count and dictionary have the same ordering
    except that dictionary has extra 'PAD' symbol at index 0

    Args:
        word_count: List of (word, count) tuples
        dict_file:  Path to the output file.
    """
    #
    dict_file = open(dict_file, 'w+')
    for word, _ in word_count:
        dict_file.write(word + '\n')
    dict_file.close()


def get_data(sentences, num_total_words, dictionaries, args):
    """ Gets data ready for training.

    Args:
        sentences:          list of training sentences
        num_total_words:    Total number of words in training corpus.
        dictionaries:       Dictionary of dictionary (urgh) and word counts.
        args:               Args passed to the script.
    """

    logger = logging.getLogger('main')
    logger.info('Building train data...')
    # Get the relevant dictionaries
    dictionary = dictionaries['dictionary']
    word_count = dictionaries['word_count']

    # If we want to use word2vec's dictionary swap here.
    # This is for debugging only, to compare with embeddings
    # generated from original word2vec.
    if args.use_w2v_weights:
        dictionary_w2v, word_count_w2v = get_w2v_dictionaries(num_total_words, args)
        # Do some sanity checks
        checks.check_word_counts(word_count, word_count_w2v)
        checks.check_dictionaries(dictionary, dictionary_w2v)
        # Swap the dictionaries
        dictionary = dictionary_w2v
        reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        word_count = word_count_w2v

    # See if we want to load pre-generated data instead of building it.
    if args.load_data:
        return np.load(args.x_file + '.npy'), np.load(args.y_file + '.npy'), np.load(args.yneg_file + '.npy')

    # Get the probabilities of keeping the words during downsampling
    keep_prob = get_keep_probs(word_count, num_total_words, args.ds_param)

    # Dump the dictionary into a file.
    save_word_counts(word_count, args.dict_file)

    # Get the training data. This returns a list of ([context], target, [negative samples]) tuples.
    train_data = get_train_data_with_sentence_downsampling(sentences, dictionaries, args)

    # Break training data into arrays of context words, targets and negative samples.
    x_train, y_train, y_neg = process_data(train_data, word_count, args)
    logger.info('Finished building train data...')

    # Dump the files to a file
    np.save(args.x_file, x_train)
    np.save(args.y_file, y_train)
    np.save(args.yneg_file, y_neg)

    return x_train, y_train, y_neg


def get_w2v_dictionaries(n_words, args):

    # For comparison purposes, tracking w2v dictionary and word_counts here as well.
    dictionary = {'PAD': 0, 'UNK': 1}
    word_count = [('UNK', 0)]
    n_known_words = 0

    # Load the dictionary and word counts from word2vec run.
    with open(args.w2v_dict_file) as vocab:
        for line in vocab:
            word, count = line.split()
            dictionary[word] = len(dictionary)
            word_count.append((word, int(count)))
            n_known_words += int(count)

    word_count[0] = ('UNK', n_words - n_known_words)

    return dictionary, word_count


def get_dictionaries(words, args):
    """ Returns a dictionary of dictionaries used in training.

    Args:
        words: A list of words from the training file.
        args:  The arguments passed on to the script.

    Returns:
        A dictionary of consisting of

        dictionary              -- dictionary mapping words to indices.
        reversed_dictionary     -- dictionary indices to words.
        word_count              -- dictionary mapping words to the number of times they occur in the corpus
        keep_prob               -- a list of probabilities of keeping them during down-sampling.
        ns_prob                 -- a list of probabilities of getting sampled during NS
    """
    logger = logging.getLogger('main')
    logger.info('Building dictionaries...')
    start_time = time.time()
    # List of (word, word_count) tuples
    word_count = [('UNK', 0)]

    # Total number of the words in the corpus
    num_total_words = len(words)

    # Sort the list of words by frequency and pick the top vocab_size ones
    if args.vocab_size == 0:
        # noinspection PyArgumentList
        # vocab_size = 0 implies we take the entire vocabulary available from the corpus
        word_count.extend(collections.Counter(words).most_common())
    else:
        # noinspection PyArgumentList
        word_count.extend(collections.Counter(words).most_common(args.vocab_size - 1))

    # Build the dictionary
    dictionary = dict()
    dictionary['PAD'] = 0
    # num_vocab_words stores the number of words in the corpus that exist in our dictionary.
    num_vocab_words = 0
    for word, count in word_count:
        num_vocab_words += count
        dictionary[word] = len(dictionary)
    # Update word count list
    word_count[0] = ('UNK', num_total_words - num_vocab_words)

    # Get the reversed dictionary.
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    # Get the negative sampling probabilities
    ns_probs = get_ns_probs(word_count, args.ns_param)

    # Get the probabilities of keeping the words during downsampling
    keep_probs = get_keep_probs(word_count, num_total_words, args.ds_param)

    dictionaries = {'dictionary': dictionary,
                    'reversed_dictionary': reversed_dictionary,
                    'word_count': word_count,
                    'ns_probs': ns_probs,
                    'keep_probs': keep_probs}

    elapsed_time = time.time() - start_time
    logger.info('Finished building dictionaries in %d seconds' % elapsed_time)
    return dictionaries


def downsample_sentence(sentence_in, dictionaries):
    """ Downsamples the training sentences exactly as in word2vec.
        * Words not in the vocabulary are omitted.
        * EOS symbols are also omitted.

    Args:
        sentence_in:    The input sentence that will be downsampled
        dictionaries:   List of dictionaries

    Returns:
        The downsampled sentence
    """

    dictionary = dictionaries['dictionary']
    keep_probs = dictionaries['keep_probs']

    sentence_out = []
    sentence_words = sentence_in.split()
    for ind, word in enumerate(sentence_words):
        # Ignore the UNK words
        if dictionary.get(word, 1) == 1:
            continue
        # Ignore the EOS word
        if word == 'EOS':
            continue
        # Sub-sample the frequent words.
        random_number = np.random.rand()
        if keep_probs.get(word) < random_number:
            continue
        sentence_out.append(word)
    return ' '.join(sentence_out)


def get_train_data_with_sentence_downsampling(sentences, dictionaries, args):
    """ This is the new implementation of get_train_data where the downsampling is done before building the context on
    each sentence. The main differences with get_train_data_with_context_downsampling implementation are
        * Downsampling is done before building context on each sentence.
        * Context window size is downsized randomly for each sentence.

    Args:
        sentences:            list of sentences in the training data
        dictionaries:         a list of dictionaries including
            dictionary:             dictionary of the vocabulary words mapping words to indices
            reversed_dictionary:    dictionary mapping indices to their corresponding words
            word_count:             a list of (word, word_count) tuples
            ns_probs:               dictionary of negative sampling probabilities
            keep_prob:              a dictionary mapping words to their probability of staying during downsampling
        args:                 input args

    Returns:
        train_data:           A list of (context, target, neg_samples) tuples
    """
    logger = logging.getLogger('main')
    train_data = []

    # Get the required dictionaries
    ns_probs = dictionaries['ns_probs']
    dictionary = dictionaries['dictionary']
    reversed_dictionary = dictionaries['reversed_dictionary']

    num_processed_sentences = 0
    num_total_sentences = len(sentences)
    logger.info('Number of sentences: %d' % num_total_sentences)
    for sentence in sentences:
        # Note that the downsampled sentence will not contain 'UNK' or 'EOS' symbols.
        sentence = downsample_sentence(sentence, dictionaries)
        sentence_words = sentence.split()
        num_processed_words = 0
        num_total_words = len(sentence_words)
        for ind, word in enumerate(sentence_words):
            # Get the dictionary index for the given word. This is our target
            # W2 matrix does not contain 'PAD' or 'UNK', so we shift the target index by two
            target_ind = dictionary.get(word) - 2
            # Build context for the current word in the sentence.
            # Shrink context window by a random number
            context_window = np.random.randint(1, args.context_window + 1)
            context = []
            for cont_ind in range(ind - context_window, ind + context_window + 1):
                if cont_ind < 0:
                    continue
                if cont_ind == ind:
                    continue
                if cont_ind >= len(sentence_words):
                    continue
                if dictionary.get(sentence_words[cont_ind], 1) == 1:
                    continue
                context.append(dictionary.get(sentence_words[cont_ind]))

            if len(context) != 0:
                # If we are doing negative sampling, build a set of negative samples
                neg_samples = []
                if args.ns_param != 0:
                    # Pick neg_samples of negative samples.
                    while len(neg_samples) < args.num_neg_samples:
                        # Pick a random word from the dictionary (ignoring 'PAD', 'UNK' and 'EOS')
                        # according to probabilities stored in ns_prob table.
                        neg_ind = np.random.choice(np.arange(2, len(dictionary)), p=ns_probs)
                        # Ignore if the random pick is the EOS symbol, or the target index
                        if reversed_dictionary.get(neg_ind) == 'EOS' \
                                or neg_ind == target_ind \
                                or neg_ind in neg_samples:
                            continue
                        # W2 matrix does not contain 'PAD' or 'UNK', so we shift the dictionary by two
                        neg_samples.append(neg_ind - 2)
                train_data.append((context, target_ind, neg_samples))

            num_processed_words += 1
            if num_processed_words % 1000 == 0:
                logger.info('Processed words for sentence: %.3f%%' % (float(num_processed_words * 100) / num_total_words))

        num_processed_sentences += 1
        if num_processed_sentences % 1000 == 0:
            logger.info('Processed sentences: %.3f%%' % (float(num_processed_sentences * 100) / num_total_sentences))

    return train_data


def process_data(train_data, word_count, args):
    # Find the size of the training examples
    M = len(train_data)
    # Get the dictionary size
    V = len(word_count)
    if args.num_neg_samples > 0:
        # We are doing negative sampling
        # x_train holds the entire training data, where each row represents context words for that training example.
        x_train = np.zeros((M, 2 * args.context_window), dtype=np.int32)
        # Each row in y_train represents the target label
        y_train = np.zeros((M, 1), dtype=np.int32)
        # each row in y_neg is a set of K negative examples for that training example.
        y_neg = np.zeros((M, args.num_neg_samples), dtype=np.int32)

        for index, (context, target, neg_samples) in enumerate(train_data):
            # Fill the corresponding column of the x_train matrix
            for cw_ind, cw in enumerate(context):
                x_train[index, cw_ind] = cw
            # Fill the corresponding column of the y_train matrix
            y_train[index, 0] = target
            # Fill the corresponding column of the y_neg matrix
            for ind, neg_sample in enumerate(neg_samples):
                y_neg[index, ind] = neg_sample
    else:
        # We are doing softmax
        # x_train holds the entire training data, where each row represents context for one training example
        x_train = np.zeros((M, V), dtype=np.float32)
        # Each column in y_train represents the one-hot encoding of the target word
        y_train = np.zeros((M, V), dtype=np.float32)
        # each column in y_neg is a set of K negative examples for that training example.
        y_neg = np.zeros((M, V), dtype=np.float32)

        for index, (context, target, neg_samples) in enumerate(train_data):
            # Fill the corresponding row of the x_train matrix
            for cw in context:
                x_train[index, cw] = 1
            # Fill the corresponding row of the y_train matrix
            y_train[index, target] = 1
            # Fill the corresponding row of the y_neg matrix
            for neg_sample in neg_samples:
                y_neg[index, neg_sample] = 1

    return x_train, y_train, y_neg


def get_ns_probs(word_count, ns_param):
    """ Returns a list of the probabilities of picking each word as a negative sample.
    List is ordered as word_count without the 'UNK' (this is not considered in any of these calculations).

    :param word_count: The dictionary containing mappings from words to their count in the corpus.
    :param ns_param:   The negative sampling parameter used when building the probability distribution.

    :return:           A list of probabilities for each word.
    """

    ns_probs = []

    # Compute normalization constant so that probabilities add up to 1.
    norm_const = 0
    for word, count in word_count[1:]:  # TODO: Think about this
        norm_const += np.power(count, ns_param)

    # Compute the probabilities for each word.
    for word, count in word_count[1:]:  # <- Skip 'UNK'
        word_prob = np.power(count, ns_param) / norm_const
        ns_probs.append(word_prob)

    return ns_probs


def get_keep_probs(word_count, num_total_words, ds_param):
    """ Returns a list of probabilities of keeping the corresponding words during downsampling

    :param word_count:          A list containing tuples of (word, word_count)
    :param num_total_words:     Total number of words in the corpus
    :param ds_param:            The downsampling parameter, used in the distribution

    :return:                    A dictionary mapping words to their probabilities
    """
    # Build the probabilities of keeping the words when downsampling
    keep_prob = []

    for word, count in word_count[1:]:  # <- Ignore 'UNK'
        # Compute the fraction of the words in the vocabulary that are the current word.
        word_frac = float(count) / num_total_words
        # Compute the probability of keeping the current word.
        word_prob = (np.sqrt(word_frac / ds_param) + 1) * ds_param / word_frac
        keep_prob.append(word_prob)

    return keep_prob


def get_mini_batches(X, Y, YNEG, batch_size=64, shuffled=True):
    """Split the data into minibatches of batch_size

    :param X:           array containing the context words at each row
    :param Y:           array containing the target word at each row
    :param YNEG:        array containing the negative samples at each row
    :param batch_size:  size of the mini-batch
    :param shuffled:    If true, training examples will be shuffled before building mini-batches

    :return:            a list of mini-batches.
    """
    logger = logging.getLogger('main')
    logger.info('Processing into mini-batches...')
    mini_batches = []
    # Get the total number of training examples
    n_training_examples = X.shape[0]

    # If shuffled=True, shuffle X and Y
    if shuffled:
        permutation = list(np.random.permutation(n_training_examples))
        X = X[permutation, :]
        Y = Y[permutation, :]
        YNEG = YNEG[permutation, :]

    num_full_batches = int(math.floor(n_training_examples / batch_size))

    for k in range(0, num_full_batches):
        mini_batch_X = X[k * batch_size: (k + 1) * batch_size, :]
        mini_batch_Y = Y[k * batch_size: (k + 1) * batch_size, :]
        mini_batch_YNEG = YNEG[k * batch_size: (k + 1) * batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y, mini_batch_YNEG)
        mini_batches.append(mini_batch)

    if n_training_examples % batch_size != 0:
        mini_batch_X = X[num_full_batches * batch_size:, :]
        mini_batch_Y = Y[num_full_batches * batch_size:, :]
        mini_batch_YNEG = YNEG[num_full_batches * batch_size:, :]
        mini_batch = (mini_batch_X, mini_batch_Y, mini_batch_YNEG)
        mini_batches.append(mini_batch)

    logger.info('Finished processing mini-batches.')

    return mini_batches


def get_negative_samples(target, num_neg_samples, dictionaries):
    neg_samples = []
    # Get the required dictionaries
    dictionary = dictionaries['dictionary']
    reversed_dictionary = dictionaries['reversed_dictionary']
    ns_probs = dictionaries['ns_probs']
    # Pick negative samples.
    # * We do not want to pick 'PAD' or 'UNK' as negative samples from the dictionary.
    # * W2 matrix does not contain 'PAD' or 'UNK' symbols (which is where we get our
    #   negative embeddings from), so our samples are shifted from the dictionary by two.
    samples = np.arange(len(dictionary) - 2)
    # Pick num_neg_samples of negative samples.
    while len(neg_samples) < num_neg_samples:
        # Pick a random word from the samples according
        # to probabilities stored in ns_prob table.
        neg_ind = np.random.choice(samples, p=ns_probs)
        # Ignore if the random pick is the target index or has already been picked
        # TODO: This is actually not strictly necessary.
        if neg_ind == target or neg_ind in neg_samples:
            continue
        neg_samples.append(neg_ind)
    # Alternatively, if we don't care about having target
    # in negative samples we could do something like this:
    # neg_samples = np.random.choice(samples, size=num_neg_samples, replace=False, p=ns_probs)
    return neg_samples


sentence_index = 0
word_index = 0


def get_training_example(sentences, dictionaries, args):
    """ Generates a single training example from the input sentences sequentially
    (a.k.a. we keep track of positioning on the sentence and the target word)

    :param sentences:         A list of sentences, where each sentence is a list of word indices
    :param dictionaries:      The dictionaries built from corpus
    :param args:              Scripts arguments

    :return:                  A tuple of ([context], target, [negative samples])
    """
    logger = logging.getLogger('main')
    global sentence_index
    global word_index

    current_sentence = sentences[sentence_index]
    target = current_sentence[word_index] - 2
    # Shrink context window by random amount
    context_window = np.random.randint(1, args.context_window + 1)
    context = []
    low = max(word_index - context_window, 0)
    high = min(word_index + context_window + 1, len(current_sentence))
    for cont_ind in range(low, high):
        # Target word cannot be part of context
        if cont_ind == word_index:
            continue
        # Do not use 'UNK' words as context
        # TODO: Remove this check if downsampling is applied
        # if current_sentence[cont_ind] == 1:
        #     continue
        context.append(current_sentence[cont_ind])

    # Pad context with zeros
    while len(context) < 2 * args.context_window:
        context.append(0)

    neg_samples = get_negative_samples(target, args.num_neg_samples, dictionaries)

    # Advance the word_index to the next word
    word_index += 1
    # If we reached the end of the sentence, advance to next sentence and reset word index
    if word_index >= len(current_sentence):
        sentence_index += 1
        word_index = 0
    # If we reached the end of the sentences, reset sentence_index back to the first one
    if sentence_index >= len(sentences):
        sentence_index = 0
        logger.info('Epoch completed.')

    return context, target, neg_samples


def get_training_batch(sentences, dictionaries, args):
    # Each row in x_train represent a context vector for a single training example
    x_train = np.zeros(shape=(args.batch_size, 2 * args.context_window), dtype=np.int32)
    # Each row in y_train represents the target label
    y_train = np.zeros(shape=(args.batch_size, 1), dtype=np.int32)
    # each row in y_neg is a set of K negative examples for that training example.
    y_neg = np.zeros(shape=(args.batch_size, args.num_neg_samples), dtype=np.int32)

    for i in range(args.batch_size):
        context, target, neg_samples = get_training_example(sentences, dictionaries, args)
        x_train[i, :] = context
        y_train[i, :] = target
        y_neg[i, :] = neg_samples

    return x_train, y_train, y_neg




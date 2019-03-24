import numpy as np
import random


class Dataset(object):
    def __init__(self, pos_data: list, neg_data: list):
        """
        Init dataset
        :param pos_data: positive data, list of positive sentences
        :param neg_data: negative data, list of negative sentences
        """
        self.pos_data = pos_data
        self.neg_data = neg_data
        self.data = self.pos_data + self.neg_data
        print('data size=%d' % len(self.data))

    def shuffle_data(self, seed: int):
        """
        :param seed: random seed
        """
        Dataset.shuffle(seed, self.pos_data)
        Dataset.shuffle(seed, self.neg_data)

    def split(self, train: float, evaluation: float, rand_seed: int = 0):
        """
        Split dataset to train_set, evaluation_set and test_set
        :param train: partition of train, in range [0, 1]
        :param evaluation: partition of evaluation, in range[0, 1]
        :param rand_seed: random seed to shuffle the data
        :return: a tuple like (test_data, test_label,
                               train_data, train_label,
                               evaluate_data, evaluate_label)
        """
        pos = Dataset._split(self.pos_data, [0, 1], train, evaluation)
        neg = Dataset._split(self.neg_data, [1, 0], train, evaluation)
        total = [a + b for a, b in zip(pos, neg)]
        total = [Dataset.shuffle(rand_seed, total[2 * i], total[2 * i + 1]) for i in range(3)]
        return total

    @staticmethod
    def _split(data: list, label, train, evaluation):
        """
        Split data with given partition and shuffle with rand_seed
        :param data: input data
        :param label: one hot vector or list
        :param train: partition of train, in range [0, 1]
        :param evaluation: partition of evaluation, in range[0, 1]
        :return: a tuple like (test_data, test_label,
                               train_data, train_label,
                               evaluate_data, evaluate_label)
        """
        if train + evaluation > 1:
            raise IndexError('sum of train and evaluation must leq 1, but get %f' % (train + evaluation))
        size = len(data)
        a, b = int(size * train), int(size * (train + evaluation))

        test_data = data[b:]
        train_data = data[:a]
        evaluate_data = data[a:b]

        test_label = [label[:] for _ in range(size - b)]
        train_label = [label[:] for _ in range(a)]
        evaluate_label = [label[:] for _ in range(b - a)]
        return test_data, test_label, train_data, train_label, evaluate_data, evaluate_label

    @staticmethod
    def shuffle(seed, data, label=None):
        """
        Shuffle data and label, work like shuffle zip of data and label
        :param seed: random seed
        :param data: input data
        :param label: corresponding label
        """
        random.Random(seed).shuffle(data)
        if label is not None:
            random.Random(seed).shuffle(label)


def batch_generator(data, batch_size: int = 32, epochs: int = 50, shuffle: bool = True):
    """
    Create a batch generator
    :param data: input data
    :param batch_size: size of batch
    :param epochs: epochs
    :param shuffle: shuffle data if True
    :return: one batch one time
    """
    data_size = len(data)
    n_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(n_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

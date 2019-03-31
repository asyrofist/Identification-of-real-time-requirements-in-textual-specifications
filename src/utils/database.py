from src.utils.TextPreProcessor import TextPreProcessor
from scipy import sparse
import pickle


class Database(object):
    example_path = '../data/temp/example_set'

    @staticmethod
    def load_example(mode):
        """
        Return previously stored features with respect to given mode

        :param mode: must be in ['tfidf', 'word2vec']
        :return: tf_idf matrix if input mode is 'tfidf'
                 pos_data, neg_data if input mode is 'word2vec', both pos_data and neg_data are set of sentences
                                                                 every sentence is a list of tokens
        """
        if mode == 'tfidf':
            return Database._load_tf_idf_example()
        elif mode == 'word2vec':
            return Database._load_word2vec_example()
        else:
            raise ValueError('mode %s not implemented please choose from ["tfidf", "word2vec"]')

    @staticmethod
    def dump_to_file(filepath_prefix: str, pos: list, neg: list, stop_words: set, embedding_type: str = 'default'):
        """
        Change input data to features and store them in [filepath + '.tfidf', filepath + '.word2vec']
        :param filepath_prefix: path prefix of dumped file
        :param pos: list of positive sentences
        :param neg: list of negative sentences
        :param stop_words: set of stop_words
        :param embedding_type: type of word embedding, must be in ['default', 'google_word2vec']
        """
        old_example_path = Database.example_path
        Database.example_path = filepath_prefix
        Database.dump_example(pos, neg, stop_words, embedding_type)
        Database.example_path = old_example_path

    @staticmethod
    def load_from_file(filepath_prefix, mode):
        old_example_path = Database.example_path
        Database.example_path = filepath_prefix
        data = Database.load_example(mode)
        Database.example_path = old_example_path
        return data

    @staticmethod
    def dump_example(pos: list, neg: list, stop_words: set, embedding_type: str = 'default'):
        """
        Change input data to features and store them in files

        :param pos: list of positive sentences
        :param neg: list of negative sentences
        :param stop_words: set of stop_words
        :param embedding_type: type of word embedding, must be in ['default', 'google_word2vec']
        """
        Database.dump_tf_idf_example(pos + neg, stop_words)
        Database.dump_word2vec_example(pos, neg, stop_words, embedding_type)

    @staticmethod
    def _load_tf_idf_example():
        """
        Load tf_idf matrix from example_file

        :return: tf_idf matrix
        :raise: IOError if cannot find previously dumped file
        """
        path = Database.example_path + '.tf_idf'
        try:
            return sparse.load_npz(path)
        except IOError as e:
            print('please dump your example first!')
            raise e

    @staticmethod
    def _load_word2vec_example():
        """
        Load word2vec features, i.e. list of sentences
                                     a sentence is a list of tokens
        The order of sentences are same to input sentence list used in dump
        :return: pos data and neg data both of which are list of sentences, every sentence is a list of tokens
        :raise: IOError if cannot find previously dumped file
        """
        path = Database.example_path + '.word2vec'
        try:
            pos, neg = pickle.load(path)
            return pos, neg
        except IOError as e:
            print('please dump your example first!')
            raise e

    @staticmethod
    def dump_tf_idf_example(corpus: list, stop_words: set):
        """
        Change corpus to feature and store features in example_file

        :param corpus: list of input sentences
        :param stop_words: set of stop words
        """
        tf_idf = TextPreProcessor.get_tf_idf_matrix(corpus, stop_words)
        path = Database.example_path + '.tf_idf'
        sparse.save_npz(path, tf_idf)

    @staticmethod
    def dump_word2vec_example(pos: list, neg: list, stop_words: set, embedding_type: str):
        """
        Change sentences to feature and store features in example_file,
        stored features can be fetched by Dataset.load_example

        :param pos: list of input positive sentences
        :param neg: list of input negative sentences
        :param stop_words: set of stop words
        :param embedding_type: type of embedding, should be 'default' or 'google_word2vec'
        """
        pos_data = []
        for sentence in pos:
            pos_data.append(TextPreProcessor.sentence2word_embeddings(sentence, stop_words, embedding_type))
        neg_data = []
        for sentence in neg:
            neg_data.append(TextPreProcessor.sentence2word_embeddings(sentence, stop_words, embedding_type))

        data = [pos_data, neg_data]
        path = Database.example_path + '.word2vec'
        pickle.dump(data, path)

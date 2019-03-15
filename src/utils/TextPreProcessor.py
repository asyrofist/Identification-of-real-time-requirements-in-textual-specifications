from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import gensim

embeddings_types = [
    'default',
    'google_word2vec',
]
google_word2vec_path = './data/word_embedding/GoogleNews-vectors-negative300.bin'


class TextPreProcessor(object):
    nlp = None
    nlp_type = None
    default_stop_words = None
    google_embeddings = None

    @staticmethod
    def get_default_stop_words():
        include_list = [
            'above', 'after', 'afterwards', 'again', 'already', 'always', 'before', 'behind', 'below', 'between',
            'beyond', 'can', 'cannot', 'could', 'last', 'then', 'under', 'until', 'up', 'upon', 'when', 'will',
        ]
        if TextPreProcessor.default_stop_words is None:
            stop_words = nlp.load('en').Defaults.stop_words
            stop_words -= include_list
            TextPreProcessor.default_stop_words = stop_words
        return TextPreProcessor.default_stop_words

    @staticmethod
    def clear_all():
        """
        Free all the resources
        """
        TextPreProcessor.nlp = None
        TextPreProcessor.nlp_type = None
        TextPreProcessor.stop_words = None
        TextPreProcessor.google_embeddings = None

    @staticmethod
    def init_nlp(embedding):
        """
        Init nlp model according to embedding type
        :param embedding: type of embedding, see embeddings_types
        """
        if embedding == 'default':
            if TextPreProcessor.nlp_type != 'large':
                TextPreProcessor.nlp_type = 'large'
                TextPreProcessor.nlp = spacy.load('en_core_web_lg')
        elif embedding == 'google_word2vec':
            if TextPreProcessor.nlp_type != 'small':
                TextPreProcessor.nlp_type = 'small'
                TextPreProcessor.nlp = spacy.load('en_core_web_sm')
        else:
            raise NotImplementedError('embedding mode {} not supported'.format(embedding))

    @staticmethod
    def get_google_word2vec(tokens: list):
        """
        add google word2vec embedding (300d) to tok as tok.vector
        :param tokens: list of tokens
        :return: list of tokens with word embeddings
        """
        if TextPreProcessor.google_embeddings is None:
            TextPreProcessor.google_embeddings = gensim.models.KeyedVectors.load_word2vec_format(google_word2vec_path,
                                                                                                 binary=True)
        for tok in tokens:
            tok.vector = TextPreProcessor.google_embeddings[tok.text]
            tok.has_vector = True
        return tokens

    @staticmethod
    def sentence2word_embeddings(sentence: str,
                                 stop_words: set,
                                 embedding_type: str = 'default'):
        """
        get word embedding of input sentence with certain mode
        :param sentence: input sentence string
        :param embedding_type: see embeddings_types
        :param stop_words: stop word list
        :return: list tokens, embedding can be fetched by tok.vector
        """
        TextPreProcessor.init_nlp(embedding_type)

        doc = TextPreProcessor.nlp(sentence)
        tokens = list(doc)
        tokens = [tok for tok in tokens if tok.text not in stop_words]

        if embeddings_types == 'google_word2vec':
            tokens = TextPreProcessor.get_google_word2vec(tokens)

        return tokens

    @staticmethod
    def get_tf_idf_matrix(corpus: list,
                          stop_words: set = None):

        """
        :param corpus: iterable container of strings
        :param stop_words: set of stop words
        :return: sparse tf-idf matrix
            :type: scipy.sparse.csr_matrix
        """

        tf_idf_vectorizer = TfidfVectorizer(stop_words=stop_words)
        tf_idf_matrix = tf_idf_vectorizer.fit_transform(corpus)
        return tf_idf_matrix

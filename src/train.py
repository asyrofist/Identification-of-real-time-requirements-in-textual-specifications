from src.models.NB import NBModel
from src.utils.api import *
from src.utils.database import Database
from src.utils.TextPreProcessor import TextPreProcessor
import time

start_time = time.time()
SIZE = 5000
EMBEDDING_TYPE = 'default'
FEATURE_MODE = 'word2vec'  # or 'tfidf', 'word2vec'
DOC_LIST = (1, 2, 4, 7, 11, 16, 19, 20, 21, 22, 25, 26, 30, 31, 32, 37, 39)
# DOC_LIST = (25,)
STOP_WORDS = TextPreProcessor.get_default_stop_words()
DIVISION = [[0, 1], [2]]
RATIO = [1, 1]
CHANGE_RATIO_MODE = 'overSample'

raw_sentences = fetch(number_of_sentences=SIZE, doc_list=DOC_LIST)
print('Total sentences: ', len(raw_sentences))

data = RatioChanger.partition(RatioChanger.get_map(raw_sentences))
pos_sentences = []
neg_sentences = []
for dtype in DIVISION[0]:
    neg_sentences += data[dtype]
for dtype in DIVISION[1]:
    pos_sentences += data[dtype]
print('Positive sentences ', len(pos_sentences))
print('Negative sentences ', len(neg_sentences))

Database.dump_example(pos_sentences, neg_sentences, STOP_WORDS, EMBEDDING_TYPE)
pos, neg = Database.load_example(FEATURE_MODE)
pos, neg = change_feature_ratio(list(pos), list(neg), RATIO, mode=CHANGE_RATIO_MODE)
print('Feature fetched! %ss' % (time.time() - start_time))
print('pos ', len(pos))
print('neg ', len(neg))


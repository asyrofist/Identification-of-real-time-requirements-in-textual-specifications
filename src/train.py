from src.models.NB import NBModel
from src.utils.api import *
from src.utils.database import Database
from src.utils.TextPreProcessor import TextPreProcessor
from src.utils.dataset import Dataset
import time
import numpy as np

start_time = time.time()
RAND_SEED = 415
SIZE = int(1E6)
EMBEDDING_TYPE = 'default'
FEATURE_MODE = 'word2vec'  # or 'tfidf', 'word2vec'
DOC_LIST = (1, 2, 4, 7, 11, 16, 19, 20, 21, 22, 25, 26, 30, 31, 32, 37, 39, 51, 53)
# DOC_LIST = (25,)
STOP_WORDS = TextPreProcessor.get_default_stop_words()
DIVISION = [[0, 1], [2]]
RATIO = [1, 1]
CHANGE_RATIO_MODE = 'overSample'

raw_sentences = fetch(number_of_sentences=SIZE, doc_list=DOC_LIST, seed=RAND_SEED)
print('Total sentences: ', len(raw_sentences))

data = RatioChanger.partition(RatioChanger.get_map(raw_sentences))
pos_sentences = []
neg_sentences = []
for dtype in DIVISION[0]:
    neg_sentences += data[dtype]
for dtype in DIVISION[1]:
    pos_sentences += data[dtype]


# print('Positive sentences ', len(pos_sentences))
# print('Negative sentences ', len(neg_sentences))


def change_inner_ratio(x, label):
    pos_label = (0, 1)
    neg_label = (1, 0)
    data = list(zip(x, label))
    pos, neg = [], []
    for v, lab in data:
        if lab == pos_label:
            pos.append((v, lab))
        elif lab == neg_label:
            neg.append((v, lab))
        else:
            raise ValueError('Unknown label', lab)
    pos, neg = change_feature_ratio(pos, neg, RATIO, mode=CHANGE_RATIO_MODE, seed=RAND_SEED)
    data = pos + neg
    random.Random(0).shuffle(data)
    x, label = list(zip(*data))
    return x, label


# Database.dump_example(pos_sentences, neg_sentences, STOP_WORDS, FEATURE_MODE)
pos, neg = Database.load_example(FEATURE_MODE)
if FEATURE_MODE == 'word2vec':
    avg = lambda sentence: sum([item.vector for item in sentence]) / len(sentence)
    pos = [avg(sentence) for sentence in pos if len(sentence)]
    neg = [avg(sentence) for sentence in neg if len(sentence)]

test_data, test_label, train_data, train_label, evaluate_data, evaluate_label = \
    Dataset(list(pos), list(neg)).split(0.6, 0.35, RAND_SEED)
test_data, test_label = change_inner_ratio(test_data, test_label)
train_data, train_label = change_inner_ratio(train_data, train_label)
evaluate_data, evaluate_label = change_inner_ratio(evaluate_data, evaluate_label)
print('Feature fetched! %ss' % (time.time() - start_time))
print('pos ', len(pos))
print('neg ', len(neg))

model_name = "Model_" + FEATURE_MODE
if FEATURE_MODE == "word2vec":
    model_name += "_" + EMBEDDING_TYPE
model_name += "_" + str(RAND_SEED)
model = NBModel(name=model_name, author='wang')
train_data = [np.array(item).reshape(-1) for item in train_data]
evaluate_data = [np.array(item).reshape(-1) for item in evaluate_data]
model.train(train_data, train_label)
evaluate = list(zip(evaluate_data, [0] * len(evaluate_label), [""] * len(evaluate_label), evaluate_label))
model.evaluate(evaluate)

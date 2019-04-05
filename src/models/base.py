import csv
import datetime
import abc


class BaseModel(object):
    value_index = 0
    doc_index = 1
    notation_index = 2
    label_index = 3
    FALSE_LABEL = (1, 0)
    TRUE_LABEL = (0, 1)
    fields = ['doc_id', 'precision', 'recall', 'accuracy', 'fbeta_score']

    def __init__(self, author, name=None, **kwargs):
        self.author = author
        self.trained = False
        self.name = name
        self.doc_list = set()
        self.data_size = 0

    def train(self, x, y, **kwargs):
        self.data_size = len(x)
        self._model_train(x, y, **kwargs)
        self.trained = True

    def get_data(self, data):
        x = []
        y = []
        for item in data:
            x.append(item[BaseModel.value_index])
            y.append(item[BaseModel.label_index])
            if item[BaseModel.doc_index] not in self.doc_list:
                self.doc_list.add(item[BaseModel.doc_index])
        return x, y

    @abc.abstractmethod
    def _estimate(self, x, y, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def _model_train(self, x, y, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def update_counter(counter, pred, label):
        if pred == 1 and label == BaseModel.TRUE_LABEL:
            counter['TP'] += 1
        if pred == 1 and label == BaseModel.FALSE_LABEL:
            counter['FP'] += 1
        if pred == 0 and label == BaseModel.TRUE_LABEL:
            counter['FN'] += 1
        if pred == 0 and label == BaseModel.FALSE_LABEL:
            counter['TN'] += 1

    def generate_filename(self):
        return "../data/temp/" + self.name + '_%s_%d_by_' % (
            datetime.datetime.now().strftime('%m_%d_%H_%M'), self.data_size) + self.author + '.csv'

    @staticmethod
    def _get_line(doc_results, doc, beta):
        counter = doc_results[doc]
        TP, FP, TN, FN = counter['TP'], counter['FP'], counter['TN'], counter['FN']
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        accuracy = (TP + TN) / (TP + FP + TN + FN)
        fbeta_score = (1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall)
        line = {BaseModel.fields[0]: doc,
                BaseModel.fields[1]: precision,
                BaseModel.fields[2]: recall,
                BaseModel.fields[3]: accuracy,
                BaseModel.fields[4]: fbeta_score}
        return line

    def evaluate(self, test_data, **kwargs):
        if not self.trained:
            raise RuntimeError('The model is not trained yet.\nEvaluation will now terminate.')

        beta = kwargs.pop("beta", 1)

        x, y = self.get_data(test_data)
        y = [yy[1] for yy in y]
        estimation, score = self._estimate(x, y, **kwargs)
        doc_results = {doc: {'TP': 1, 'FP': 1, 'TN': 1, 'FN': 1}
                       for doc in self.doc_list}

        for pred, item in zip(estimation, test_data):
            counter = doc_results[item[BaseModel.doc_index]]
            BaseModel.update_counter(counter, pred, item[BaseModel.label_index])

        with open(self.generate_filename(), 'wt', newline="") as outfile:
            output = csv.DictWriter(outfile, BaseModel.fields)
            output.writeheader()
            for doc in self.doc_list:
                line = BaseModel._get_line(doc_results, doc, beta)
                output.writerow(line)
        return doc_results, score

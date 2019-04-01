import csv
import datetime
import abc


class BaseModel(object):
    value_index = 0
    doc_index = 1
    notation_index = 2
    label_index = 3
    fields = ['doc_id', 'precision', 'recall', 'accuracy']

    def __init__(self, author, name=None, **kwargs):
        self.author = author
        self.trained = False
        self.name = name
        self.doc_list = set()
        self.data_size = 0

    def train(self, data, **kwargs):
        self.data_size += len(data)
        x, y = self.get_data(data)
        self._model_train(x, y, **kwargs)

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
    def _estimate(self, x, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def _model_train(self, x, y, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def update_counter(counter, pred, label):
        if pred == 1 and label == 1:
            counter['TP'] += 1
        if pred == 1 and label == 0:
            counter['FP'] += 1
        if pred == 0 and label == 1:
            counter['FN'] += 1
        if pred == 0 and label == 0:
            counter['TN'] += 1

    def generate_filename(self):
        return self.name + '_%s_%d_by_' % (
            datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'), self.data_size) + self.author + '.csv'

    @staticmethod
    def _get_line(doc_results, doc):
        counter = doc_results[doc]
        TP, FP, TN, FN = counter['TP'], counter['FP'], counter['TN'], counter['FN']
        line = {BaseModel.fields[0]: doc,
                BaseModel.fields[1]: TP / (TP + FP),
                BaseModel.fields[2]: TP / (TP + FN),
                BaseModel.fields[3]: (TP + TN) / (TP + FP + TN + FN)}
        return line

    def evaluate(self, test_data, **kwargs):
        if not self.trained:
            raise RuntimeError('The model is not trained yet.\nEvaluation will now terminate.')

        x, _ = self.get_data(test_data)
        estimation = self._estimate(x, **kwargs)
        doc_results = {doc: {'TP': 1, 'FP': 1, 'TN': 1, 'FN': 1}
                       for doc in self.doc_list}

        for pred, item in zip(estimation, test_data):
            counter = doc_results[item[BaseModel.doc_index]]
            BaseModel.update_counter(counter, pred, item[BaseModel.label_index])

        with open(self.generate_filename(), 'wt') as outfile:
            output = csv.DictWriter(outfile, BaseModel.fields)
            for doc in self.doc_list:
                line = BaseModel._get_line(doc_results, doc)
                output.writerow(line)
        return doc_results

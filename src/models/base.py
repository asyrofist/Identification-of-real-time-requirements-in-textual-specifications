import csv
import datetime
import abc


class BaseModel(object):
    Y_ID = 3

    def __init__(self, author=None, name=None, **kwargs):
        if author is None:
            print("Warning: Author is not specified. Continue?(Y/n)", end='')
            ret = input()
            if "Y" not in ret and "y" not in ret:
                exit(1)
            self.author = "Anonymous"
        else:
            self.author = author
        self.trained = False
        self.name = name
        self.start_time = datetime.datetime(1900, 1, 1)
        self.doc_list = set()
        self.result = {}
        self.data_size = 0

    def train(self, data, **kwargs):
        self.data_size += len(data)
        X = []
        y = []
        for item in data:
            X.append(item[0])
            y.append(item[BaseModel.Y_ID])
            if item[1] not in self.doc_list:
                self.doc_list.add(item[1])
        self._model_train(X, y, **kwargs)

    @abc.abstractmethod
    def _estimate(self, X, **kwargs):
        pass

    @abc.abstractmethod
    def _model_train(self, X, y, **kwargs):
        pass

    def evaluate(self, test_data, **kwargs):
        self.start_time = datetime.datetime.now()
        if not self.trained:
            print("Unable to evaluate.The model is not trained yet.")
            print("Evaluation will now terminate.")
            return
        output_filename = self.name + "_%s_%d_by_" % (
            self.start_time.strftime("%Y_%m_%d_%H_%M_%S"), self.data_size) + self.author + ".csv"

        X = []
        for item in test_data:
            X.append(item[0])
            if item[1] not in self.doc_list:
                self.doc_list.add(item[1])

        estimation = self._estimate(X, **kwargs)

        self.result = {}
        for doc in self.doc_list:
            self.result[doc] = {"true positive": 0,
                                "false positive": 0,
                                "true negative": 0,
                                "false negative": 0}
        for id, item in enumerate(test_data):
            if estimation[id] == 1 and item[BaseModel.Y_ID] == 1:
                self.result[item[1]]['true positive'] += 1
            if estimation[id] == 1 and item[BaseModel.Y_ID] == 0:
                self.result[item[1]]['false positive'] += 1
            if estimation[id] == 0 and item[BaseModel.Y_ID] == 1:
                self.result[item[1]]['false negative'] += 1
            if estimation[id] == 0 and item[BaseModel.Y_ID] == 0:
                self.result[item[1]]['true negative'] += 1
        fields = ['doc_id', 'precision', 'recall', 'accuracy']
        with open(output_filename, "wt") as outfile:
            output = csv.DictWriter(outfile, fields)
            for doc in self.doc_list:
                line = {"doc_id": doc,
                        "precision": self.result[doc]['true positive'] / (
                            self.result[doc]['true positive'] + self.result[doc]['false positive'] + 1),
                        "recall": self.result[doc]['true positive'] / (
                            self.result[doc]['true positive'] + self.result[doc]['false negative'] + 1),
                        "accuracy": (self.result[doc]['true positive'] + self.result[doc]['true negtive']) / (
                            self.result[doc]['true positive'] + self.result[doc]['false positive']
                            + self.result[doc]['true negative'] + self.result[doc]['false negative'])
                        }
                output.writerow(line)
        return self.result

from .base import BaseModel
import sklearn.neighbors.classification as knn


class KNNModel(BaseModel):
    def __init__(self, author=None, name=None, **kwargs):
        super().__init__(author=author, name=name, **kwargs)

        self.model = knn.KNeighborsClassifier()

    def _model_train(self, x, y, **kwargs):
        self.model.fit(x, y)

    def _estimate(self, x, **kwargs):
        result = self.model.predict(x)
        return result.tolist()

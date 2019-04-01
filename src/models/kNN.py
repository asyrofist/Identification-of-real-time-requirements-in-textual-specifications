from base import BaseModel
import sklearn.neighbors.classification as kNN


class KNNModel(BaseModel):
    model_type_list = {
        "kNN": kNN.KNeighborsClassifier
    }

    def __init__(self, author=None, name=None, type="Gaussian", **kwargs):
        super.__init__(self, author=author, name=name, **kwargs)

        self.model = KNNModel.model_type_list[type](**kwargs)

    def _model_train(self, X, y, **kwargs):
        partial = kwargs.get("partial", False)
        if partial:
            print("kNN does not support partial training, overwrite the model instead")
        self.model.fit(X, y)

    def _estimate(self,X,**kwargs):
        result = self.model.predict(X)
        return result.tolist()

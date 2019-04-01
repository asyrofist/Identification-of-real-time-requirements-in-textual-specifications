from base import BaseModel
import sklearn.svm as svm


class SVMModel(BaseModel):
    model_type_list = {
        "SVM": svm.SVC,
        "nuSVM":svm.NuSVC
    }

    def __init__(self, author=None, name=None, type="SVM", **kwargs):
        super.__init__(self, author=author, name=name, **kwargs)

        self.model = SVMModel.model_type_list[type](**kwargs)

    def _model_train(self, X, y, **kwargs):
        partial = kwargs.get("partial", False)
        if partial:
            print("SVM does not support partial training, overwrite the model instead")
        self.model.fit(X, y)

    def _estimate(self,X,**kwargs):
        result = self.model.predict(X)
        return result.tolist()

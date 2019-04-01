from .base import BaseModel
import sklearn.svm as svm


class SVMModel(BaseModel):
    model_type_list = {
        "SVM": svm.SVC,
        "nuSVM": svm.NuSVC
    }

    def __init__(self, author=None, name=None, model_type="SVM", **kwargs):
        super().__init__(author=author, name=name, **kwargs)

        self.model = SVMModel.model_type_list[model_type](**kwargs)

    def _model_train(self, x, y, **kwargs):
        self.model.fit(x, y)

    def _estimate(self, x, **kwargs):
        result = self.model.predict(x)
        return result.tolist()

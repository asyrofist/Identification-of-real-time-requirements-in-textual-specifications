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
        y = [yy[1] - yy[0] for yy in y]
        self.model.fit(x, y)

    def _estimate(self, x, y, **kwargs):
        result = self.model.predict(x)
        score = self.model.score(x, y)
        return result.tolist(), score

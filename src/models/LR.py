from .base import BaseModel
import sklearn.linear_model as lr


class LR(BaseModel):
    model_type_list = {
        "naive": lr.LogisticRegression,
        "CV": lr.LogisticRegressionCV,
    }

    def __init__(self, author=None, name=None, model_type="naive", **kwargs):
        super().__init__(author=author, name=name, **kwargs)

        self.model = LR.model_type_list[model_type](**kwargs)

    def _model_train(self, x, y, **kwargs):
        y = [yy[1] for yy in y]
        self.model.fit(x, y)

    def _estimate(self, x, y, **kwargs):
        result = self.model.predict(x)
        score = self.model.score(x, y)
        return result.tolist(), score

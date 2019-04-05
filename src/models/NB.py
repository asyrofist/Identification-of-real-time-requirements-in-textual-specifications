from .base import BaseModel
import sklearn.naive_bayes as nb


class NB(BaseModel):
    def __init__(self, author=None, name=None, **kwargs):
        super().__init__(author=author, name=name, **kwargs)

        self.model = nb.GaussianNB()

    def _model_train(self, x, y, **kwargs):
        y = [yy[1] for yy in y]
        partial = kwargs.pop("partial", False)
        if partial:
            self.model.partial_fit(x, y, **kwargs)
        else:
            self.model.fit(x, y, **kwargs)

    def _estimate(self, x, y, **kwargs):
        result = self.model.predict(x)
        score = self.model.score(x, y)
        return result.tolist(), score

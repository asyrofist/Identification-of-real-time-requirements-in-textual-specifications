from base import BaseModel
import sklearn.naive_bayes as nb


class NBModel(BaseModel):
    def __init__(self, author=None, name=None,**kwargs):
        super.__init__(self,author=author, name=name,**kwargs)

        self.model = nb.GaussianNB()

    def _model_train(self, X, y, **kwargs):
        partial = kwargs.pop("partial",False)
        if partial:
            self.model.partial_fit(X,y,**kwargs)
        else:
            self.model.fit(X,y,**kwargs)

    def _estimate(self,X,**kwargs):
        result = self.model.predict(X)
        return result.tolist()

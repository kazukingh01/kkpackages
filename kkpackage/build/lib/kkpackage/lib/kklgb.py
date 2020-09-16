import lightgbm as lgb
# local package
from kkpackage.util.learning import softmax, sigmoid


class KkLGBMClassifier(lgb.LGBMClassifier):
    """
    custom objective を想定して値を規格化できるように KkLGBMClassifier の自作classを定義する
    下手にinit 処理しない
    """
    def predict_proba(self, X, *argv, **kwargs):
        """
        値を規格化するため override する
        """
        proba = super().predict_proba(X, *argv, **kwargs)
        if len(proba.shape) == 2:
            proba = softmax(proba)
        else:
            proba = sigmoid(proba)
            proba[:, 0] = 1 - proba[:, 1]
        return proba
    
    def rm_objective(self):
        """
        lambda x: で定義した function は消さないと pickle 化できない
        """
        self.objective  = None
        self._objective = None
        self._fobj      = None
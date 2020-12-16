from sklearn.calibration import CalibratedClassifierCV


__all__ = [
    "Calibrater",
]


class _MockCalibrater:
    def __init__(self, classes):
        self.classes_ = classes
    def predict_proba(self, X):
        return X
    def __str__(self):
        return "MockCalibrater"


class Calibrater:
    """
    CalibratedClassifierCVは交差検証時にValidaionデータでfittingを行う
    本クラスでは独自に交差検証を実装しているため、交差するのが面倒くさい
    なので、入力X(predict_proba)に対して、そのままpredict_probaが帰ってくるような
    擬似sklearnクラスを自作する
    """
    def __init__(self, model):
        """
        Params::
            model: Fitting済みのmodel
        """
        self.model    = model
        self.classes_ = self.model.classes_
        self.mock_calibrater = _MockCalibrater(self.model.classes_)
        self.calibrater      = CalibratedClassifierCV(self.mock_calibrater, cv="prefit", method='isotonic')

    def __str__(self):
        return str(self.calibrater)

    def fit(self, X, Y, **kwargs):
        """
        ここで入力するXはpredict_proba である. 実際の特徴量ではない点注意
        """
        self.calibrater.fit(X, Y, **kwargs)

    def predict_proba_mock(self, X, **kwargs):
        """
        ここで入力するXはpredict_proba である. 実際の特徴量ではない点注意
        """
        return self.calibrater.predict_proba(X)

    def predict_proba(self, X, **kwargs):
        """
        ここで入力するXは実際の特徴量である
        """
        return self.calibrater.predict_proba(self.model.predict_proba(X))
        
    def predict(self, X, **kwargs):
        """
        ここで入力するXは実際の特徴量である
        """
        return self.calibrater.predict(self.model.predict_proba(X))


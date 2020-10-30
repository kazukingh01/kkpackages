import numpy as np
import pandas as pd
from typing import List
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

# local package
from kkpackage.util.dataframe import nanmap
from kkpackage.util.common import check_type, is_callable
from kkpackage.util.logger import set_logger
logger = set_logger(__name__)



class ProcRegistry(object):
    def __init__(self, colname_explain: np.ndarray, colname_answer: np.ndarray):
        super().__init__()
        self.processing = {}
        self.default_proc(colname_explain, colname_answer)


    def default_proc(self, colname_explain: np.ndarray, colname_answer: np.ndarray):
        logger.info("START")
        check_type(colname_explain, [np.ndarray])
        check_type(colname_answer,  [np.ndarray])
        self.processing["default_x"] = {}
        self.processing["default_x"]["type"] = "x"
        self.processing["default_x"]["cols"] = colname_explain
        self.processing["default_x"]["proc"] = []
        self.processing["default_y"] = {}
        self.processing["default_y"]["type"] = "y"
        self.processing["default_y"]["cols"] = colname_answer
        self.processing["default_y"]["proc"] = []
        self.processing["default_row"] = {}
        self.processing["default_row"]["type"] = "row"
        self.processing["default_row"]["cols"] = None
        self.processing["default_row"]["proc"] = []
        logger.info("END")


    def __call__(self, df: pd.DataFrame, autofix: bool=False, x_proc: bool=True, y_proc: bool=True, row_proc: bool=True):
        logger.info("START")
        # row proc
        if row_proc: df = self.proc_row(df)
        # col proc
        list_x, list_y = [], []
        for name in self.processing.keys():
            if self.processing[name]["type"] not in ["x", "y"]: continue
            if x_proc == False and self.processing[name]["type"] == "x": continue
            if y_proc == False and self.processing[name]["type"] == "y": continue
            logger.info(f'name: {name}, type: {self.processing[name]["type"]}')
            if len(self.processing[name]["proc"]) > 0 and isinstance(self.processing[name]["proc"][0], MyAsType):
                ## 始めの処理がAsTypeの場合はメモリの軽減のためにndf前に適応する
                ndf = df[self.processing[name]["cols"]].astype(self.processing[name]["proc"][0].convert_type).values.copy()
            else:
                ndf = df[self.processing[name]["cols"]].values.copy()
            for _proc in self.processing[name]["proc"]:
                logger.info(f'proc: {_proc}')
                shape_before = ndf.shape
                logger.info(f"before shape: {ndf.shape}")
                ndf = _proc(ndf)
                logger.info(f"after  shape: {ndf.shape}")
                if shape_before[0] != ndf.shape[0]:
                    logger.raise_error("The number of rows is different from before and after process.")
            if   self.processing[name]["type"] == "x": list_x.append(ndf)
            elif self.processing[name]["type"] == "y": list_y.append(ndf)
        if autofix:
            if len(list_x) == 1: list_x = list_x[0]
            if len(list_y) == 1: list_y = list_y[0]
        logger.info(f"after processing x: \n{list_x}")
        logger.info(f"after processing y: \n{list_y}")
        logger.info("END")
        return list_x, list_y
    

    def proc_row(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("START")
        df = df.copy()
        for name in self.processing.keys():
            if self.processing[name]["type"] not in ["row"]: continue
            logger.info(f'name: {name}, type: {self.processing[name]["type"]}')
            for _proc in self.processing[name]["proc"]:
                logger.info(f'proc: {_proc}')
                logger.info(f"before shape: {df.shape}")
                df = _proc(df)
                logger.info(f"after  shape: {df.shape}")
        logger.info("END")
        return df


    def register(self, list_proc: list, name: str=None, type_proc: str=None, columns: np.ndarray=None):
        """
        処理を登録する. 登録名が新規の場合は新たにprocessingを登録する
        Params::
            list_proc: callable な関数やクラスのリスト
            name: 登録処理名
            type_proc: "x" or "y". rowの場合、default_rowに追加するのみとする
        """
        logger.info("START")
        if name is None and isinstance(type_proc, str) and type_proc == "x":   name = "default_x"
        if name is None and isinstance(type_proc, str) and type_proc == "y":   name = "default_y"
        if name is None and isinstance(type_proc, str) and type_proc == "row": name = "default_row"
        if name not in list(self.processing.keys()):
            if type_proc not in ["x", "y"]:
                logger.raise_error(f'type_proc must be "x" or "y". type_proc: {type_proc}')
            self.processing[name] = {}
            self.processing[name]["type"] = type_proc
            self.processing[name]["cols"] = columns
            self.processing[name]["proc"] = []
        for _proc in list_proc:
            self.processing[name]["proc"].append(_proc)
        logger.info("END")
    

    def set_columns(self, columns: np.ndarray, name: str=None, type_proc: str=None):
        """
        説明変数を再度セットする
        Params::
            columns: 新規の説明変数
            name: None の場合, type_proc が指定されていれば、defaultに自動セットする
            type_proc: x or y
        """
        if name is None and isinstance(type_proc, str) and type_proc == "x": name = "default_x"
        if name is None and isinstance(type_proc, str) and type_proc == "y": name = "default_y"
        self.processing[name]["cols"] = columns
    

    def fit(self, df: pd.DataFrame):
        """
        登録した処理に関するパラメータを学習するため、入力データを基準にfittingさせる
        """
        logger.info("START")
        df = self.proc_row(df) # row proc
        for name in self.processing.keys():
            if self.processing[name]["type"] not in ["x", "y"]: continue
            logger.info(f'name: {name}, type: {self.processing[name]["type"]}')
            if len(self.processing[name]["proc"]) > 0 and isinstance(self.processing[name]["proc"][0], MyAsType):
                ## 始めの処理がAsTypeの場合はメモリの軽減のためにndf前に適応する
                ndf = df[self.processing[name]["cols"]].astype(self.processing[name]["proc"][0].convert_type).values.copy()
            else:
                ndf = df[self.processing[name]["cols"]].values
            for _proc in self.processing[name]["proc"]:
                logger.info(f'proc: {_proc}')
                if is_callable(_proc, "fit"):
                    # Fitting
                    _proc.fit(ndf)
                logger.info(f"before shape: {ndf.shape}")
                ndf = _proc(ndf) # fit後、適用する
                logger.info(f"after  shape: {ndf.shape}")
        logger.info("END")



class MyMinMaxScaler(MinMaxScaler):
    def __str__(self): return self.__class__.__name__
    def __call__(self, ndf: np.ndarray):
        return self.transform(ndf)

class MyStandardScaler(StandardScaler):
    def __str__(self): return self.__class__.__name__
    def __call__(self, ndf: np.ndarray):
        return self.transform(ndf)

class MyPCA(object):
    def __init__(self, pca_cutoff: float=0.99):
        self.model = PCA(n_components=pca_cutoff)
    def __str__(self): return self.__class__.__name__
    def __call__(self, ndf):
        return self.model.transform(ndf)
    def fit(self, ndf: np.ndarray):
        self.model.fit(ndf)

class MyFillNa(object):
    def __init__(self, fill_value: object):
        """
        Params::
            fill_value: 埋める値. int, float, object. mean, max, min, median の場合は入力に応じて決める
        """
        self.fill_value = fill_value
        self.fit_values = None
    def __str__(self):
        return f'{self.__class__.__name__}(fill_value: {self.fill_value})'
    def __call__(self, ndf: np.ndarray):
        ndf = ndf.copy()
        if (type(self.fill_value) == str):
            # 列毎に埋める値が異なる場合は、列毎にループして欠損補完する
            for i in np.arange(ndf.shape[1]):
                ndf[:, i][np.isnan(ndf[:, i])] = self.fit_values[i]
        else:
            ndf[np.isnan(ndf)] = self.fill_value
        return ndf
    def fit(self, ndf: np.ndarray):
        if   (type(self.fill_value) == str) and (self.fill_value == "mean"):
            self.fit_values = np.nanmean(ndf, axis=0).copy()
        elif (type(self.fill_value) == str) and (self.fill_value == "max"):
            self.fit_values = np.nanmax(ndf, axis=0).copy()
        elif (type(self.fill_value) == str) and (self.fill_value == "min"):
            self.fit_values = np.nanmin(ndf, axis=0).copy()
        elif (type(self.fill_value) == str) and (self.fill_value == "median"):
            self.fit_values = np.nanmedian(ndf, axis=0).copy()
        else:
            # self.fill_value の値で埋める
            pass

class MyFillNaRandom(object):
    def __init__(self, bins: int=100):
        self.bins = bins
        self.hist = []
    def __str__(self):
        return f'{self.__class__.__name__}(bins: {self.bins})'
    def __call__(self, ndf: np.ndarray):
        is_nan = np.isnan(ndf)
        n_nan  = is_nan.sum(axis=0)
        for i, (bins, dens) in enumerate(self.hist):
            if n_nan[i] == 0: continue
            choice = np.random.choice(bins, n_nan[i], p=dens)
            ndf[is_nan[:, i], i] = choice
        return ndf
    def fit(self, ndf: np.ndarray):
        self.hist = [np.histogram(ndf[:, i][~np.isnan(ndf[:, i])], bins=self.bins) for i in np.arange(ndf.shape[1])]
        self.hist = [(np.array([bins[j:j+2].mean() for j in np.arange(self.bins)]), (dens / dens.sum())) for dens, bins in self.hist]

class MyFillNaMinMax(object):
    def __init__(self, add_value: float=1):
        self.add_value  = add_value
        self.min = np.zeros(0)
        self.max = np.zeros(0)
    def __str__(self):
        return f'{self.__class__.__name__}()'
    def __call__(self, ndf: np.ndarray):
        is_nan = np.isnan(ndf)
        n_nan  = is_nan.sum(axis=0)
        for i, (_min, _max) in enumerate(zip(self.min, self.max)):
            if n_nan[i] == 0: continue
            choice = np.random.choice([_min, _max], n_nan[i])
            ndf[is_nan[:, i], i] = choice
        return ndf
    def fit(self, ndf: np.ndarray):
        self.min = np.nanmin(ndf, axis=0) - self.add_value
        self.max = np.nanmax(ndf, axis=0) + self.add_value

class MyReplaceValue:
    def __init__(self, target_value: object, replace_value: object):
        self.target_value  = target_value
        self.replace_value = replace_value
    def __str__(self):
        return f'{self.__class__.__name__}(target_value: {self.target_value}, replace_value: {self.replace_value})'
    def __call__(self, ndf: np.ndarray):
        ndf = ndf.copy()
        ndf[ndf == self.target_value] = self.replace_value
        return ndf

class MyOneHotEncoder:
    def __init__(self, target_indexes: List[int]):
        check_type(target_indexes, [list, tuple])
        self.target_indexes = target_indexes
        self.model = OneHotEncoder(categories='auto')
    def __str__(self): return self.__class__.__name__
    def __call__(self, ndf: np.ndarray):
        bool_col = np.zeros(ndf.shape[1]).astype(bool)
        bool_col[self.target_indexes] = True
        output = self.model.transform(ndf[:, bool_col].copy()).toarray()
        ndf = ndf[:, ~bool_col].copy()
        ndf = np.concatenate([ndf, output], axis=1)
        return ndf
    def fit(self, ndf: np.ndarray):
        self.model.fit(ndf)

class MyAsType:
    """ Classを定義しないとpickle化できない """
    def __init__(self, convert_type: object):
        self.convert_type = convert_type
    def __str__(self):
        return f'{self.__class__.__name__}(convert_type: {self.convert_type})'
    def __call__(self, ndf: np.ndarray):
        ndf = ndf.copy().astype(self.convert_type)
        return ndf

class MyReshape:
    """ Classを定義しないとpickle化できない """
    def __init__(self, convert_shape: tuple):
        self.convert_shape = convert_shape if type(convert_shape) in [list, tuple] else (convert_shape, )
    def __str__(self):
        return f'{self.__class__.__name__}(convert_shape: {self.convert_shape})'
    def __call__(self, ndf: np.ndarray):
        ndf = ndf.copy().reshape(*self.convert_shape)
        return ndf

class MyDropNa:
    def __init__(self, columns: np.ndarray):
        self.columns = columns
    def __str__(self):
        return f'{self.__class__.__name__}(columns: {self.columns})'
    def __call__(self, df: pd.DataFrame):
        boolwk = np.zeros(df.shape[0]).astype(bool)
        for x in self.columns:
            boolwk = (boolwk | df[x].isna().values)
        return df.loc[~boolwk, :]

class MyCondition:
    def __init__(self, colname: str, condition: str, value: object):
        if not isinstance(colname, str): logger.raise_error(f'colname: {colname} is not string.')
        if condition not in [">", "<", "=", ">=", "<="]: logger.raise_error(f'condition: {condition} is not expected value.')
        self.colname   = colname
        self.condition = condition
        self.value     = value
    def __str__(self):
        return f'{self.__class__.__name__}(colname: {self.colname}, condition: {self.condition}, value: {self.value})'
    def __call__(self, df: pd.DataFrame):
        if   self.condition == "=":
            df = df[df[self.colname] == self.value]
        elif self.condition == ">":
            df = df[df[self.colname] >  self.value]
        elif self.condition == ">=":
            df = df[df[self.colname] >= self.value]
        elif self.condition == "<":
            df = df[df[self.colname] <  self.value]
        elif self.condition == "<=":
            df = df[df[self.colname] <= self.value]
        return df

class MyDictMap:
    def __init__(self, columns: np.ndarray, dict_values: dict):
        self.columns     = columns
        self.dict_values = dict_values
    def __str__(self):
        return f'{self.__class__.__name__}(columns: {self.columns}, dict_values: {self.dict_values})'
    def __call__(self, df: pd.DataFrame):
        df = df.copy()
        for x in self.columns:
            df[x] = nanmap(df[x], self.dict_values)
        return df

class Calibrater:
    """
    CalibratedClassifierCVは交差検証時にValidaionデータでfittingを行う
    本クラスでは独自に交差検証を実装しているため、交差するのが面倒くさい
    なので、入力X(predict_proba)に対して、そのままpredict_probaが帰ってくるような
    擬似sklearnクラスを自作する
    """
    class _MockCalibrater:
        def __init__(self, classes):
            self.classes_ = classes
        def predict_proba(self, X):
            return X
        def __str__(self):
            return "MockCalibrater"

    def __init__(self, model):
        """
        Params::
            model: Fitting済みのmodel
        """
        self.model    = model
        self.classes_ = self.model.classes_
        self.mock_calibrater = self._MockCalibrater(self.model.classes_)
        self.calibrater      = CalibratedClassifierCV(self.mock_calibrater, cv="prefit", method='isotonic')

    def __str__(self):
        return str(self.calibrater)

    def fit(self, X, Y):
        """
        ここで入力するXはpredict_proba である. 実際の特徴量ではない点注意
        """
        self.calibrater.fit(X, Y)

    def predict_proba_mock(self, X):
        """
        ここで入力するXはpredict_proba である. 実際の特徴量ではない点注意
        """
        return self.calibrater.predict_proba(X)

    def predict_proba(self, X):
        """
        ここで入力するXは実際の特徴量である
        """
        return self.calibrater.predict_proba(self.model.predict_proba(X))
        
    def predict(self, X):
        """
        ここで入力するXは実際の特徴量である
        """
        return self.calibrater.predict(self.model.predict_proba(X))


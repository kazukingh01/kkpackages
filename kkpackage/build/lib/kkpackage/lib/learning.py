import numpy as np
import pandas as pd
from typing import List
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

# local package
from kkpackage.util.dataframe import nanmap, conv_ndarray
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
        if row_proc:
            df = self.proc_row(df)
            if x_proc == False and y_proc == False:
                ## df だけの返却方法
                return df
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
                    logger.warning("The number of rows is different from before and after process.")
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
        df = df.copy()
        for name in self.processing.keys():
            if self.processing[name]["type"] not in ["row"]: continue
            logger.info(f'name: {name}, type: {self.processing[name]["type"]}')
            for _proc in self.processing[name]["proc"]:
                logger.info(f'proc: {_proc}')
                logger.info(f"before shape: {df.shape}")
                if is_callable(_proc, "fit"):
                    # Fitting
                    _proc.fit(df)
                df = _proc(df)
                logger.info(f"after  shape: {df.shape}")
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

class MyRankGauss(QuantileTransformer):
    """ output_distribution='normal' とすれば Gauss分布に変換. uniform は一様分布"""
    def __init__(self, *args, columns: np.ndarray=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.columns = conv_ndarray(columns) if columns is not None else None
    def __str__(self): return f'{self.__class__.__name__}(columns: {self.columns})'
    def __call__(self, ndf: np.ndarray):
        if self.columns is None and isinstance(ndf, np.ndarray):
            return self.transform(ndf)
        elif isinstance(self.columns, np.ndarray) and isinstance(ndf, pd.DataFrame):
            ndf[self.columns] = self.transform(ndf[self.columns].values.copy())
            return ndf
        else:
            logger.raise_error(f'input is not expected type. {self.columns}, {ndf}')
    def fit(self, X, **kwargs):
        if self.columns is None and isinstance(X, np.ndarray):
            super().fit(X, **kwargs)
        elif isinstance(self.columns, np.ndarray) and isinstance(X, pd.DataFrame):
            super().fit(X[self.columns].values, **kwargs)
        else:
            logger.raise_error(f'input is not expected type. {self.columns}, {X}')


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
        bool_col = np.zeros(ndf.shape[1]).astype(bool)
        bool_col[self.target_indexes] = True
        self.model.fit(ndf[:, bool_col])

class MyOneHotInverse:
    def __init__(self, target_indexes: List[int]):
        check_type(target_indexes, [list, tuple])
        self.target_indexes = target_indexes
    def __str__(self): return f'{self.__class__.__name__}(target_indexes: {self.target_indexes})'
    def __call__(self, ndf: np.ndarray):
        bool_col = np.zeros(ndf.shape[1]).astype(bool)
        bool_col[self.target_indexes] = True
        indexes = np.where(ndf[:, bool_col].astype(bool))[1].copy()
        if indexes.shape[0] != ndf.shape[0]:
            logger.raise_error(f'input values is not OneHot type: \n{ndf}')
        ndf = ndf[:, ~bool_col].copy()
        ndf = np.concatenate([ndf, indexes.reshape(-1, 1)], axis=1)
        return ndf
    def fit(self, ndf: np.ndarray):
        bool_col = np.zeros(ndf.shape[1]).astype(bool)
        bool_col[self.target_indexes] = True
        if (ndf[:, bool_col].sum(axis=1) == 1).sum() != ndf.shape[0]:
            logger.raise_error(f'input values is not OneHot type: \n{ndf}')

class MyOneHotInverseUnique:
    def __init__(self, target_indexes: List[int]=None):
        check_type(target_indexes, [list, tuple, type(None)])
        self.target_indexes = target_indexes
        self.dict_conv = {}
    def __str__(self): return f'{self.__class__.__name__}(target_indexes: {self.target_indexes})'
    def __call__(self, ndf: np.ndarray):
        if self.target_indexes is None:
            se = pd.DataFrame(ndf).apply(lambda x: str(x.tolist()),axis=1).astype(str)
            ndf = se.map(self.dict_conv).values
        else:
            bool_col = np.zeros(ndf.shape[1]).astype(bool)
            bool_col[self.target_indexes] = True
            ndfwk = ndf[self.target_indexes]
            ndf   = ndf[:, ~bool_col].copy()
            se = pd.DataFrame(ndfwk).apply(lambda x: str(x.tolist()),axis=1).astype(str)
            ndfwk = se.map(self.dict_conv).values
            ndf = np.concatenate([ndf, ndfwk.reshape(-1, 1)], axis=1)
        return ndf
    def fit(self, ndf: np.ndarray):
        ndfwk = None
        if self.target_indexes is None:
            ndfwk = np.unique(ndf, axis=0)
        else:
            ndfwk = np.unique(ndf[self.target_indexes], axis=0)
        self.dict_conv = {str(x.tolist()):i for i, x in enumerate(ndfwk)}

class MyOneHotAuto:
    def __init__(self, n_size_first: int=-1, n_unique_first: int=5, n_unique_second: int=5, min_unique: int=2):
        self.n_size_first    = n_size_first
        self.n_unique_first  = n_unique_first
        self.n_unique_second = n_unique_second
        self.min_unique      = min_unique
        self.dict_convert    = {}
    def __str__(self):
        return f'{self.__class__.__name__}(n_size_first: {self.n_size_first}, n_unique_first: {self.n_unique_first}, n_unique_second: {self.n_unique_second}, min_unique: {self.min_unique})'
    def __call__(self, ndf: np.ndarray):
        logger.info(f'convert: \n{self.dict_convert}')
        indexes = list(self.dict_convert.keys())
        ndfbool = np.zeros(ndf.shape[1]).astype(bool)
        ndfbool[indexes] = True
        ndfwk = ndf[:,  ndfbool].copy()
        ndf   = ndf[:, ~ndfbool]
        for i, _key in enumerate(indexes):
            for _, y in self.dict_convert[_key].items():
                ndfwk[:, i] == y
                ndf = np.concatenate([ndf, (ndfwk[:, i] == y).reshape(-1, 1)], axis=1)
        return ndf
    def fit(self, ndf: np.ndarray):
        ndf = np.sort(ndf, axis=0)
        ndfbool = None
        if self.n_size_first < 1:
            ndfbool = np.ones(ndf.shape[1]).astype(bool)
        else:
            ndfwk    = np.zeros(ndf.shape[1]).astype(int)
            ndfwk[:] = self.n_unique_first + 1
            for i in np.arange(ndf.shape[1]):
                ndfwk[i] = np.unique(ndf[:self.n_size_first, i]).shape[0]
            ndfbool = (ndfwk < self.n_unique_first) & (ndfwk > self.min_unique)
        ndfwk    = np.zeros(ndf.shape[1]).astype(int)
        ndfwk[:] = self.n_unique_second + 1
        for i in np.where(ndfbool)[0]:
            ndfwk[i] = np.unique(ndf[:, i]).shape[0]
        ndfbool = (ndfwk < self.n_unique_second) & (ndfwk > self.min_unique)
        for i in np.where(ndfbool)[0]:
            self.dict_convert[i] = {}
            for j, k in enumerate(np.sort(np.unique(ndf[:, i]))):
                self.dict_convert[i][j] = k

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
        reshape = np.array(self.convert_shape).astype(object)
        reshape[reshape == "shape0"] = ndf.shape[0]
        reshape[reshape == "shape1"] = ndf.shape[1]
        reshape = tuple(reshape.astype(int).tolist())
        ndf = ndf.copy().reshape(*reshape)
        return ndf

class MySelectIndex:
    def __init__(self, dim: int, i_shape: int, index: int):
        self.dim = dim
        self.i_shape = i_shape
        self.index = index
    def __str__(self):
        return f'{self.__class__.__name__}(dim: {self.dim}, i_shape: {self.i_shape}, index: {self.index})'
    def __call__(self, ndf: np.ndarray):
        if   self.dim == 1:
            if self.i_shape == 0:
                ndf = ndf[self.index]
            else:
                logger.raise_error(f'ndf shape: {ndf.shape} is mismatch !! (dim: {self.dim}, i_shape: {self.i_shape}, index: {self.index})')
        elif self.dim == 2:
            if   self.i_shape == 0:
                ndf = ndf[self.index, :]
            elif self.i_shape == 1:
                ndf = ndf[:, self.index]
            else:
                logger.raise_error(f'ndf shape: {ndf.shape} is mismatch !! (dim: {self.dim}, i_shape: {self.i_shape}, index: {self.index})')
        elif self.dim == 3:
            if   self.i_shape == 0:
                ndf = ndf[self.index, :, :]
            elif self.i_shape == 1:
                ndf = ndf[:, self.index, :]
            elif self.i_shape == 1:
                ndf = ndf[:, :, self.index]
            else:
                logger.raise_error(f'ndf shape: {ndf.shape} is mismatch !! (dim: {self.dim}, i_shape: {self.i_shape}, index: {self.index})')
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


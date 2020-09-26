import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA

# local package
from kkpackage.util.common import check_type, is_callable
from kkpackage.util.logger import set_logger
logger = set_logger(__name__)



class ProcRegistry(object):
    def __inti__(self, colname_explain: np.ndarray, colname_answer: np.ndarray):
        super().__init__()
        self.colname_explain = colname_explain
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
        logger.info("END")


    def __call__(self, df: pd.DataFrame):
        logger.info("START")
        list_x, list_y = [], []
        for name in self.processing.keys():
            logger.info(f'name: {name}, type: {self.processing[name]["type"]}')
            ndf = df[self.processing[name]["cols"]].values.copy()
            for _proc in self.processing[name]["proc"]:
                ndf = _proc(ndf)
            if   self.processing[name]["type"] == "x": list_x.append(ndf)
            elif self.processing[name]["type"] == "y": list_y.append(ndf)
        logger.info("END")
        return list_x, list_y


    def register(self, list_proc: list, name: str, type_proc: str=None):
        """
        処理を登録する. 登録名が新規の場合は新たにprocessingを登録する
        Params::
            list_proc: callable な関数やクラスのリスト
            name: 登録処理名
            type_proc: "x" or "y"
        """
        logger.info("START")
        if name not in list(self.processing.keys()):
            if type_proc not in ["x", "y"]:
                logger.raise_error(f'type_proc must be "x" or "y". type_proc: {type_proc}')
            self.processing[name] = {}
            self.processing[name]["type"] = type_proc
            self.processing[name]["proc"] = []
        for _proc in list_proc:
            self.processing[name]["proc"].append(_proc)
        logger.info("END")
    

    def fit(self, df: pd.DataFrame):
        """
        登録した処理に関するパラメータを学習するため、入力データを基準にfittingさせる
        """
        logger.info("START")
        for name in self.processing.keys():
            logger.info(f'name: {name}, type: {self.processing[name]["type"]}')
            ndf = df[self.processing[name]["cols"]].values.copy()
            for _proc in self.processing[name]["proc"]:
                if is_callable(_proc, "fit"):
                    # Fitting
                    _proc.fit(ndf)
                logger.info(f"before shape: {ndf.shape}")
                ndf = _proc(ndf) # fit後、適用する
                logger.info(f"after  shape: {ndf.shape}")
        logger.info("END")


class MyMinMaxScaler(MinMaxScaler):
    def __call__(self, ndf: np.ndarray):
        return self.transform(ndf)

class MyStandardScaler(StandardScaler):
    def __call__(self, ndf: np.ndarray):
        return self.transform(ndf)

class MyPCA(object):
    def __init__(self, pca_cutoff: float=0.99):
        self.model = PCA(n_components=pca_cutoff)
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

class MyReplaceValue:
    def __init__(self, target_value: object, replace_value: object):
        self.target_value  = target_value
        self.replace_value = replace_value
    def __calll__(self, ndf: np.ndarray):
        ndf = ndf.copy()
        ndf[ndf == self.target_value] = self.replace_value
        return ndf

class MyOneHotEncoder:
    def __init__(self, model):
        self.model = model
    def transform(self, X):
        # Xはnumpy形式の想定
        ndf = X.copy()
        ndf = self.model.transform(ndf).toarray()
        return ndf
    def pre_proc_one_hot_encoder(self, df, target="y"):
        self.logger.info("START")
        self.logger.info("Regist pre_proc_ No.%s, df shape:%s", len(self.preprocessing_name[target]), df.shape)

        # 登録されている処理があれば先に実行する
        ndf = self.preprocessing(df, target=target)
        # 処理の登録とFit
        ohe = OneHotEncoder(categories='auto')
        ohe.fit(ndf)
        mypreproc = self.MyOneHotEncoder(ohe)
        self.preprocessing_name[ target].append("OneHotEncoder()")
        self.preprocessing_model[target].append(mypreproc)
        self.logger.info("END")




    def preprocessing(self, df: pd.DataFrame, target: str="x"):
        """
        pre proc に登録された処理を逐次的に行う
        基本的に特徴量数やレコード数の変動は考慮しない。値の変換を想定する
        """
        self.logger.info("START")
        self.logger.info(f"df shape:{df.shape}, target={target}")
        self.logger.info(f"features:{self.colname_explain.shape}")

        ndf = None
        # 初期処理として下記を実行
        if self.preprocessing_init_col.get(target) is None:
            self.logger.raise_error(f"unexpected target:{target} !!")

        init_col = self.preprocessing_init_col[target]
        self.logger.info(f"pre processing {target} No.-1: convert to numpy value")
        ## ここで当初float32に強制変換していたが、その場合categorical変数にしたいint型も
        ## float型になり、またnumpyの全体の型がfloatだとintの列は作成できないため、
        ## catboostのcategorycalが使えなかった。しかし型がバラバラでobject型に変換されると
        ## どうしてもメモリが圧迫される。そのため登録処理の最初が型変換の場合は、この時点で処理させる。
        ndf = None
        if (len(self.preprocessing_name[target]) > 0) and \
           (self.preprocessing_name[target][0].find("ConvertCulumnType") == 0) and \
           (self.preprocessing_model[target][0].target_indexes is None):
            if type(init_col) == str:
                ndf = df[self.__getattribute__(init_col)].copy() \
                      .astype(self.preprocessing_model[target][0].convert_type).values
            else:
                ndf = df[init_col].copy() \
                      .astype(self.preprocessing_model[target][0].convert_type).values
        else:
            # 以前はobjectに強制変換していたが、objectの場合
            # np.isnan(ndf)のような全体に対しての変換ができないためやめる。
            # dfに様々な型が混在していた場合、valuesで自然とobject型になる
            if type(init_col) == str:
                ndf = df[self.__getattribute__(init_col)].copy().values
            else:
                ndf = df[init_col].copy().values
        
        # 1次元の場合はreshapeする
        if len(ndf.shape) == 1:
            ndf = ndf.reshape(-1, 1)
        
        # 登録された処理を最初から実行する(np.float32, inf の変換はここで)
        ## 上記でastypeされてももう一度実行する
        for i, _proc in enumerate(self.preprocessing_model[target]):
            self.logger.info(f"pre processing {target} No.{target}: {self.preprocessing_name[target][i]}, shape from:{ndf.shape}, type: {ndf.dtype}")
            ndf = _proc.transform(ndf)
            self.logger.info(f"pre processing {target} No.{target}: {self.preprocessing_name[target][i]}, shape from:{ndf.shape}, type: {ndf.dtype}")
        self.logger.info("END")
        return ndf


    def add_preprocessing_target(self, target: str, target_type: str, init_col):
        """
        別口で入力Xを作成する場合の関数
        """
        self.logger.info("START")
        self.logger.info(f"add target. name:{target}, type:{target_type}, init_col:{init_col}")
        # 新規に登録できるかチェックする
        if target in list(self.preprocessing_name.keys()):
            self.logger.raise_error("preproc name is already registerd !! Chage name.")

        self.preprocessing_name[ target] = [] #処理名
        self.preprocessing_model[target] = [] #処理内容
        if target_type in ["x","y"]:
            self.preprocessing_addlist.append(tuple([target_type, target]))
        else:
            # x, y以外の登録はNG.
            self.logger.raise_error("target is 'x' or 'y' !!")
        # 処理対象初期カラム名(strの場合はself.__getattr__()で取得)
        if (type(init_col) == str) or (init_col == np.ndarray):
            self.preprocessing_init_col[target] = init_col
        else:
            self.logger.raise_error("init_col's type is str or numpy !!")

        self.logger.info("END")








    def ndf_apply_preproc(self, df: pd.DataFrame, x_proc: bool=True, y_proc: bool=True) -> List[np.ndarray]:
        """
        登録されたpre_proc_の処理を実行して特徴量と正解ラベルのndarrayを作成する
        Params::
            df: input
            y_proc: 正解ラベル側のprocを処理するかどうか
        """
        X = self.preprocessing(df, target="x") if x_proc else None
        Y = self.preprocessing(df, target="y") if y_proc else None

        # 個別に追加定義したtargetがあれば作成する
        X_add, Y_add = [], []
        for _type, _name in self.preprocessing_addlist:
            ndf = self.preprocessing(df, target=_name)
            # preproc が x か y なのかで分ける
            if   _type == "x" and x_proc: X_add.append(ndf.copy())
            elif _type == "y" and y_proc: Y_add.append(ndf.copy())

        # _add があればくっつける
        X = [X, *X_add]
        Y = [Y, *Y_add]

        return X, Y


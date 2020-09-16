from typing import List, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import optuna

from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# local package
from kkpackage.util.learning import search_features_by_variance, search_features_by_correlation, \
    split_data_balance, predict_detail, evalate, eval_classification_model, eval_regressor_model, \
    is_classification_model, conv_validdata_in_fitparmas, calc_randomtree_importance
from kkpackage.util.hyperparams import search_hyperparams_by_optuna
from kkpackage.util.dataframe import conv_ndarray
from kkpackage.util.common import is_callable, correct_dirpath, makedirs, save_pickle, load_pickle

# logger
from kkpackage.util.logger import set_logger
logger = set_logger()
_logname = __name__

class MyModel:
    
    # コンストラクタ
    def __init__(
        self, name: str, 
        # model parameter
        colname_explain: np.ndarray, colname_answer: str, colname_other: np.ndarray = None,
        # model
        model=None, 
        # common parameter
        random_seed: int=1, n_jobs: int=1, log_level :str="info"
    ):
        self.logger = set_logger(_logname + "." + name, log_level=log_level, internal_log=True)
        self.logger.debug("START")

        self.name        = name
        self.random_seed = random_seed
        self.n_jobs      = n_jobs
        self.colname_explain_first = conv_ndarray(colname_explain)
        self.colname_explain       = conv_ndarray(colname_explain)
        self.colname_explain_hist  = []
        self.colname_answer        = colname_answer
        self.colname_other         = conv_ndarray(colname_other) if colname_other is not None else np.array([])
        self.feature_importances   = pd.DataFrame()
        self.feature_importances_randomtrees = pd.DataFrame()
        self.feature_importances_modeling    = pd.DataFrame()
        self.model          = model
        self.calibrater     = None
        self.is_calibration = False
        self.is_model_fit = False
        self.preprocessing_name     = {}
        self.preprocessing_model    = {}
        self.preprocessing_init_col = {}
        self.preprocessing_addlist  = []
        self.preprocessing_name[    "x"] = []
        self.preprocessing_model[   "x"] = []
        self.preprocessing_name[    "y"] = []
        self.preprocessing_model[   "y"] = []
        self.preprocessing_init_col["x"] = "colname_explain"
        self.preprocessing_init_col["y"] = "colname_answer"
        self.n_trained_samples = {}
        self.n_tested_samples  = {}
        self.index_train = np.array([]) # 実際に残す際はマルチインデックスかもしれないので、numpy形式にはならない
        self.index_valid = np.array([]) # 実際に残す際はマルチインデックスかもしれないので、numpy形式にはならない
        self.index_test = np.array([]) # 実際に残す際はマルチインデックスかもしれないので、numpy形式にはならない
        self.df_pred_train = pd.DataFrame()
        self.df_pred_valid = pd.DataFrame()
        self.df_pred_test  = pd.DataFrame()
        self.eval_train_cm = pd.DataFrame()
        self.eval_valid_cm = pd.DataFrame()
        self.eval_test_cm  = pd.DataFrame()
        self.eval_train_val = pd.Series(dtype=object)
        self.eval_valid_val = pd.Series(dtype=object)
        self.eval_test_val  = pd.Series(dtype=object)
        self.optuna_study   = None
        self.fig = {}
        self.logger.info("create instance. name:"+name)
        # model が set されていれば初期化しておく
        if model is not None: self.set_model(model)
        self.logger.debug("END")
        

    def __del__(self):
        pass


    def set_model(self, model, **params):
        """
        モデルのセット(モデルに関わる箇所は初期化)
        Params::
            model: model
            colname_explain: 特徴量のカラム名リスト
            colname_answer: 正解ラベルのカラム名
            colname_other: 交差検証時などに予測に付与したいカラム名リスト
            **params: その他インスタンスに追加したい変数
        """
        self.logger.debug("START")
        self.model = model
        self.optuna_study          = None
        self.calibrater            = None
        self.is_calibration        = False
        self.is_model_fit          = False
        for param in params.keys():
            # 追加のpreprocessingを扱う場合など
            self.__setattr__(param, params.get(param))
        self.logger.info(f'set model: \n{self.model}')
        self.logger.debug("END")
    

    def is_classification_model(self) -> bool:
        """
        Return:: 分類モデルの場合はTrueを返却する
        """
        return is_classification_model(self.model)


    def cut_features_by_variance(self, df: pd.DataFrame, cutoff: float=0.99, ignore_nan: bool=False):
        """
        各特徴量の値の重複度合いで特徴量をカットする
        Params::
            df: input DataFrame. 既に対象の特徴量だけに絞られた状態でinputする
            cutoff: 重複度合い. 0.99の場合、全体の数値の内99%が同じ値であればカットする
            ignore_nan: nanも重複度合いの一部とするかどうか
        """
        self.logger.info("START")
        self.logger.info(f"df shape:{df.shape}, cutoff:{cutoff}, ignore_nan:{ignore_nan}")
        self.logger.info(f"features:{self.colname_explain.shape}", )
        df = df[self.colname_explain] # 関数が入れ子に成りすぎてメモリが膨大になっている
        _columns = search_features_by_variance(df, cutoff=cutoff, ignore_nan=ignore_nan, n_jobs=self.n_jobs)
        # 特徴量の更新
        self.colname_explain_hist.append(self.colname_explain.copy())
        self.colname_explain = self.colname_explain[~np.isin(self.colname_explain, _columns)] 
        self.logger.info(f"cut   features by variance :{_columns.shape[0]            }. features...{_columns}")
        self.logger.info(f"alive features by variance :{self.colname_explain.shape[0]}. features...{self.colname_explain}")
        self.logger.info("END")


    def cut_features_by_correlation(self, df, cutoff=0.9, ignore_nan_mode=0, on_gpu_size=1):
        """
        相関係数の高い値の特徴量をカットする
        ※欠損無視(ignore_nan=True)しての計算は計算量が多くなるので注意
        Params::
            cutoff: 相関係数の閾値
            ignore_nan_mode: 計算するモード
                0: np.corrcoef で計算. nan がないならこれで良い
                1: np.corrcoef で計算. nan は平均値で埋める
                2: pandas で計算する. nan は無視して計算するが遅いので並列化している
                3: GPUを使って高速に計算する. nanは無視して計算する
            on_gpu_size: ignore_nan_mode=3のときに使う. 行列が全てGPUに乗り切らないときに、何分割するかの数字
        """
        self.logger.info("START")
        self.logger.info(f"df shape:{df.shape}, cutoff:{cutoff}, ignore_nan_mode:{ignore_nan_mode}")
        self.logger.info(f"features:{self.colname_explain.shape}")

        df_corr, cut_list = search_features_by_correlation(df[self.colname_explain], cutoff=cutoff, ignore_nan_mode=ignore_nan_mode, on_gpu_size=on_gpu_size, n_jobs=self.n_jobs)
        self.correlation = df_corr.copy()

        # 特徴量の更新
        self.colname_explain_hist.append(self.colname_explain.copy())
        alive_features, cut_list = self.features_by_correlation(cutoff)
        self.colname_explain = alive_features
        self.logger.info("cut   features by correlation:%s. features...%s", len(cut_list), cut_list[:10])
        self.logger.info("alive features by correlation:%s. features...%s", self.colname_explain.shape[0], self.colname_explain)
        self.logger.info("END")


    def cut_features_by_random_tree_importance(self, df: pd.DataFrame, cut_ratio: float, calc_randomtrees: bool=False, **kwargs):
        self.logger.info("START")
        self.logger.info("cut_ratio:%s", cut_ratio)
        if self.model is None:
            self.logger.raise_error("model is not set !!")
        if calc_randomtrees:
            self.feature_importances_randomtrees = calc_randomtree_importance(
                df, colname_explain=self.colname_explain, colname_answer=self.colname_answer, 
                is_cls_model=self.is_classification_model(), n_jobs=self.n_jobs, **kwargs
            )
        alive_features, cut_list = self.features_by_random_tree_importance(cut_ratio)
        self.colname_explain = alive_features
        self.logger.info("cut   features by randomtree importance:%s. features...%s", len(cut_list), cut_list[:10])
        self.logger.info("alive features by randomtree importance:%s. features...%s", self.colname_explain.shape[0], self.colname_explain)
        self.logger.info("END")


    def features_by_correlation(self, cutoff: float) -> (np.ndarray, np.ndarray):
        self.logger.info("START")
        columns = self.correlation.columns.values.copy()
        alive_features = columns[(((self.correlation > cutoff) | (self.correlation < -1*cutoff)).sum(axis=0) == 0)].copy()
        cut_list       = columns[~np.isin(columns, alive_features)]
        self.logger.info("END")
        return alive_features, cut_list


    def features_by_random_tree_importance(self, cut_ratio: float) -> (np.ndarray, np.ndarray):
        self.logger.info("START")
        self.logger.info("cut_ratio:%s", cut_ratio)
        if self.model is None:
            self.logger.raise_error("model is not set !!")
        if self.feature_importances_randomtrees.shape[0] == 0:
            self.logger.raise_error("feature_importances_randomtrees is None. You should do calc_randomtree_importance() first !!")
        _n = int(self.feature_importances_randomtrees.shape[0] * cut_ratio)
        alive_features = self.feature_importances_randomtrees.iloc[:-1*_n ]["feature_name"].values.copy()
        cut_list       = self.feature_importances_randomtrees.iloc[ -1*_n:]["feature_name"].values.copy()
        self.logger.info("END")
        return alive_features, cut_list


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


    # 正規化(最小値 ～ 最大値)
    def pre_proc_min_max_scaler(self, df: pd.DataFrame, feature_range: (int, int) = (0, 1), target :str="x"):
        self.logger.info("START")
        self.logger.info("Regist pre_proc_ No.%s, df shape:%s", \
                         len(self.preprocessing_name[target]), df.shape)

        # 登録されている処理があれば先に実行する
        ndf = self.preprocessing(df, target=target)
        # 処理の登録とFit
        mm = MinMaxScaler(feature_range=feature_range)
        mm.fit(ndf)
        self.preprocessing_name[ target].append("MinMaxScaler(feature_range="+str(feature_range)+")")
        self.preprocessing_model[target].append(mm)
        self.logger.info("END")
        

    # 標準化
    def pre_proc_standard_scaler(self, df: pd.DataFrame, target: str="x"):
        self.logger.info("START")
        self.logger.info("Regist pre_proc_ No.%s, df shape:%s", \
                         len(self.preprocessing_name[target]), df.shape)

        # 登録されている処理があれば先に実行する
        ndf = self.preprocessing(df, target=target)
        # 処理の登録とFit
        ss = StandardScaler()
        ss.fit(ndf)
        self.preprocessing_name[ target].append("StandardScaler()")
        self.preprocessing_model[target].append(ss)
        self.logger.info("END")


    # PCAで次元削減(次元が変化するので注意)
    def pre_proc_pca(self, df: pd.DataFrame, pca_cutoff: float=0.99, target: str="x"):
        self.logger.info("START")
        self.logger.info("Regist pre_proc_ No.%s, df shape:%s, cutoff:%s", \
                         len(self.preprocessing_name[target]), df.shape, pca_cutoff)

        # まずは標準化(既にしていてももう一回する)
        self.pre_proc_standard_scaler(df)
        
        # 登録されている処理があれば先に実行する
        ndf = self.preprocessing(df, target=target)
        # 処理の登録とFit
        pca = PCA(n_components=pca_cutoff)
        pca.fit(ndf)
        self.preprocessing_name[ target].append("PCA(n_components=)"+str(pca_cutoff))
        self.preprocessing_model[target].append(pca)

        self.logger.info("END")


    # 穴埋め処理
    class MyFillNa:
        def __init__(self, fill_value):
            self.fill_value = fill_value
            self.fit_values = None
        def fit(self, X):
            if   (type(self.fill_value) == str) and (self.fill_value == "mean"):
                self.fit_values = np.nanmean(X, axis=0).copy()
            elif (type(self.fill_value) == str) and (self.fill_value == "max"):
                self.fit_values = np.nanmax(X, axis=0).copy()
            elif (type(self.fill_value) == str) and (self.fill_value == "min"):
                self.fit_values = np.nanmin(X, axis=0).copy()
            else:
                pass
        def transform(self, X):
            # Xはnumpy形式の想定
            ndf = X.copy()
            if (type(self.fill_value) == str):
                # 縦列毎にループして欠損補完する
                for i in np.arange(ndf.shape[1]):
                    ndf[:, i][np.isnan(ndf[:, i])] = self.fit_values[i]
            else:
                ndf[np.isnan(ndf)] = self.fill_value
            return ndf
    def pre_proc_fillna(self, df: pd.DataFrame, fill_value, target: str="x"):
        self.logger.info("START")
        self.logger.info("Regist pre_proc_ No.%s, df shape:%s, fill_value:%s", \
                         len(self.preprocessing_name[target]), df.shape, fill_value)

        # 登録されている処理があれば先に実行する
        ndf = self.preprocessing(df, target=target)
        # 処理の登録
        mypreproc = self.MyFillNa(fill_value)
        mypreproc.fit(ndf)
        self.preprocessing_name[ target].append("FillNa, fill_value="+str(fill_value))
        self.preprocessing_model[target].append(mypreproc)
        self.logger.info("END")


    # 置き換え処理
    class MyReplaceValue:
        def __init__(self, target_value, replace_value):
            self.target_value  = target_value
            self.replace_value = replace_value
        def transform(self, X):
            # Xはnumpy形式の想定
            ndf = X.copy()
            ndf[ndf == self.target_value] = self.replace_value
            return ndf
    def pre_proc_replace_value(self, df: pd.DataFrame, target_value, replace_value, target: str="x"):
        self.logger.info("START")
        self.logger.info("Regist pre_proc_ No.%s, df shape:%s, target_value:%s, replace_value:%s", \
                         len(self.preprocessing_name[target]), df.shape, target_value, replace_value)

        # 処理の登録
        mypreproc = self.MyReplaceValue(target_value, replace_value)
        self.preprocessing_name[ target].append("ReplaceVal, target_value="+str(target_value)+\
                                                ", replace_value="+str(replace_value))
        self.preprocessing_model[target].append(mypreproc)
        self.logger.info("END")


    # 型変換処理
    class MyConvertCulumnType:
        def __init__(self, convert_type, target_indexes=None):
            self.convert_type   = convert_type
            self.target_indexes = target_indexes
        def transform(self, X):
            # Xはnumpy形式の想定
            ndf = X.copy()
            if self.target_indexes is None:
                ndf = ndf.astype(self.convert_type)
            else:
                if ndf.dtype != np.object: raise TypeError
                ndf[:, self.target_indexes] = ndf[:, self.target_indexes].astype(self.convert_type)
            return ndf
    def pre_proc_convert_culumn_type(self, df: pd.DataFrame, convert_type, target_indexes=None, target: str="x"):
        self.logger.info("START")
        self.logger.info("Regist pre_proc_ No.%s, df shape:%s, convert_type:%s, target_indexes:%s", \
                         len(self.preprocessing_name[target]), df.shape, convert_type, target_indexes)

        # target_indexesが文字列で来た場合はnumpyでのindexに変換する
        # indexは数値か数値のリストを要求する
        if target_indexes is not None:
            if   type(target_indexes) == int:
                pass
            elif type(target_indexes) == str:
                self.logger.raise_error(f"target_indexes: {target_indexes} is not match type.", TypeError)
            elif (type(target_indexes) == list) or (type(target_indexes) == np.ndarray):
                for _x in target_indexes:
                    if type(_x) == str: self.logger.raise_error(f"target_indexes: {target_indexes} is not match type.", TypeError)
                    else: int(str(_x)) # floatだったらエラーになる
            else:
                self.logger.raise_error(f"target_indexes: {target_indexes} is not match type.", TypeError)

        # 処理の登録
        mypreproc = self.MyConvertCulumnType(convert_type, target_indexes)
        self.preprocessing_name[ target].append("ConvertCulumnType, convert_type="+str(convert_type)+\
                                                ", target_indexes="+str(target_indexes))
        self.preprocessing_model[target].append(mypreproc)
        self.logger.info("END")


    # サイズ変換処理
    class MyReshape:
        def __init__(self, reshape):
            self.reshape = reshape
        def transform(self, X):
            # Xはnumpy形式の想定
            ndf = X.copy()
            if (type(self.reshape) == type(np.array([]))):
                if   len(self.reshape.shape) == 1:
                    ndf = ndf[:, self.reshape]
                elif len(self.reshape.shape) == 2:
                    # RNN用に変換する. DataFrameを経由して変換させる
                    df = pd.DataFrame(index=np.arange(ndf.shape[0]))
                    for i, index in enumerate(self.reshape):
                        dfwk  = pd.DataFrame(ndf[:, index])
                        df[i] = dfwk.apply(lambda x: x.values,axis=1).copy()
                    # 左列から順に時系列順になっている.DataFrameでapplyしてvstackする
                    ## この時点で各レコードには、[[a,b,c,..], [a,b,c,..], [a,b,c,..],..]の二次元配列化される
                    df = df.apply(lambda x: np.vstack(x), axis=1)
                    # さらに全体をvstackしてreshapeする
                    ndf = np.vstack(df.values)
                    ndf = ndf.reshape(X.shape[0], self.reshape.shape[0], \
                                      self.reshape.shape[1]) # データ数×時系列数×特徴量数
            else:
                ndf = ndf.reshape(self.reshape)
            return ndf
        def fit(self, X):
            # 変換可能かチェックする
            if (type(self.reshape) == type(np.array([]))):
                # numpy形式の場合、RNN用時系列データへの変換を意味する
                # さらにその中身も配列で縦列のインデックス(int型)が入っていることを想定する
                # [[1,2,3,4,5],
                #  [2,3,4,5,6],
                #  [3,4,5,6,7]] みたいな
                # 1次元[1,2,3,4,5]の場合は、その列を抽出する役割とする
                if len(self.reshape.shape) > 2:
                    print("mismatch type !!")
                    raise TypeError
                elif (self.reshape.dtype != np.int):
                    print("mismatch type !!")
                    raise TypeError
                elif not ((self.reshape.max() < X.shape[1]) and (self.reshape.min() >= 0)):
                    print("mismatch type !!")
                    raise TypeError
            elif (type(self.reshape) == type(())):
                # tuple の場合. 普通にreshapeに突っ込む
                pass
            elif (type(self.reshape) == int) and (self.reshape == -1):
                # -1 の場合. 普通にreshapeに突っ込む
                pass
            else:
                print("mismatch type !!")
                raise TypeError
    def pre_proc_reshape(self, df: pd.DataFrame, reshape, target="y"):
        self.logger.info("START")
        self.logger.info("Regist pre_proc_ No.%s, df shape:%s, reshape:%s", \
                         len(self.preprocessing_name[target]), df.shape, reshape)

        # 登録されている処理があれば先に実行する
        ndf = self.preprocessing(df, target=target)
        # 処理の登録
        mypreproc = self.MyReshape(reshape)
        mypreproc.fit(ndf) # エラーがあるかチェックする
        self.preprocessing_name[ target].append("Reshape, reshape="+str(reshape))
        self.preprocessing_model[target].append(mypreproc)
        self.logger.info("END")


    # One Hot Encoder. 主に正解ラベルに対して
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


    def eval_model(self, df_score, **eval_params) -> (pd.DataFrame, pd.Series, ):
        """
        回帰や分類を意識せずに評価値を出力する
        """
        df_conf, se_eval = pd.DataFrame(), pd.Series()
        if self.is_classification_model():
            df_conf, se_eval = eval_classification_model(df_score, "answer", "predict", ["predict_proba_"+str(int(_x)) for _x in self.model.classes_], labels=self.model.classes_, **eval_params)
        else:
            df_conf, se_eval = eval_regressor_model(     df_score, "answer", "predict", **eval_params)
        return df_conf, se_eval


    def fit(
            self, df_train, 
            split_params={"n_splits":1, "y_type":"cls", "weight":"balance", "is_bootstrap":False}, 
            fit_params={}, 
            eval_params={"n_round":3, "eval_auc_dict":{}}, 
            pred_params={"do_estimators":False}
        ):
        """
        モデルのfit関数. fitだけでなく、評価値や重要度も格納する
        """
        self.logger.info("START")
        self.logger.info(f"df shape:{df_train.shape}, split_params:{split_params}, fit_params:{fit_params}, eval_params:{eval_params}")
        self.logger.info(f"features:{self.colname_explain.shape}")
        if self.model is None: self.logger.raise_error("model is not set.")

        split_params = split_params.copy() if split_params is not None else {}
        fit_params   = fit_params.  copy() if fit_params   is not None else {}
        eval_params  = eval_params. copy() if eval_params  is not None else {}
        pred_params  = pred_params. copy() if pred_params  is not None else {}
        split_params["random_seed"] = self.random_seed

        # numpyに変換(前処理の結果を反映させる)
        X_train, Y_train = self.ndf_apply_preproc(df_train)

        ## データのスプリット
        split_params["random_seed"] = self.random_seed
        train_indexes, _ = split_data_balance(Y_train[0], **split_params)

        # 学習
        for _i in range(len(X_train)): X_train[_i] = X_train[_i][train_indexes[0]]
        for _i in range(len(Y_train)): Y_train[_i] = Y_train[_i][train_indexes[0]]
        ## 各データが１種類の場合は、元に戻す
        if len(X_train) == 1: X_train = X_train[0]
        else:                 X_train = tuple(X_train)
        if len(Y_train) == 1: Y_train = Y_train[0]
        else:                 Y_train = tuple(Y_train)
        self.logger.debug("create model by All samples : start ...")
        ## 個別定義したX_train_addに関して、空[]であれば通常通り(X_train, Y_train)で受け渡される
        ## ここで y=*** をつけないと、*X_train_add の所が流動的になりすぎるため。y=*** を指定することで、
        ## 目的変数の指定までは追加の説明変数であることを意識させる.
        ## scikit-learnでのfitは y=*** とは指定されていないが、y=*** とこっちで指定しても
        ## 変数の数が問題なければエラーなく通る
        ## 例) def test(x, y)
        ##  test(*x, y=y)    # xは1配列のtuple, yは通常のnumpyとか
        ##  ↑問題なく y に代入される
        ## (新)色々あったが、上記は無視してtupleならtupleでそのまま渡す
        self.model.fit(X_train, Y_train, **fit_params)
        if self.is_classification_model():
            # np.bincount(Y_train.astype(int)) だと マイナスの値でエーが出る
            sewk = pd.DataFrame(Y_train.astype(int)).groupby(0).size().sort_index()
            if is_callable(self.model, "classes_") == False:
                ## classes_ がない場合は手動で追加する
                self.model.classes_ = np.sort(np.unique(Y_train)).astype(int)
            self.n_trained_samples = {int(x):sewk[int(x)] for x in self.model.classes_}
        self.logger.debug("create model by All samples : end ...")

        # 訓練データの精度を記録
        pred_params["n_jobs"] = self.n_jobs
        df_score = predict_detail(self.model, X_train, **pred_params)
        df_score["answer"] = Y_train
        df_score["index"]  = train_indexes[0]
        for x in self.colname_other: df_score["other_"+x] = df_train.iloc[train_indexes[0], df_train.columns.get_indexer([x]).min()].copy().values
        self.index_train   = df_train.index[train_indexes[0]].copy() # DataFrame のインデックスを残しておく
        self.df_pred_train = df_score.copy()
        
        ## 評価する
        df_conf, se_eval = self.eval_model(df_score, **eval_params)
        self.logger.info("evaluation model by train data.")
        
        self.eval_train_cm  = df_conf.copy()
        self.eval_train_val = se_eval.astype(str).copy()
        self.logger.info("\n%s", self.eval_train_cm)
        self.logger.info("\n%s", self.eval_train_val)

        # 特徴量の重要度
        self.logger.info("feature importance saving...")
        if is_callable(self.model, "feature_importances_") == True:
            _df = pd.DataFrame(np.array([self.colname_explain, self.model.feature_importances_]).T, columns=["feature_name","importance"])
            _df = _df.sort_values(by="importance", ascending=False).reset_index(drop=True)
            self.feature_importances = _df.copy()

        self.is_model_fit = True
        self.logger.info("END")


    # セットしたモデルで交差検証
    ## _fit_params で _validation_x, _validation_y が記載された場合は
    ## Validation での 検証分の値を渡すこととする
    def fit_cross_validation(
            self, df_train: pd.DataFrame, 
            save_traindata: bool=False, 
            conv_autostop: [str, str]= None, 
            split_params={"n_splits":1, "y_type":"cls", "weight":"balance", "is_bootstrap":False}, 
            fit_params={}, fit_params_train={},
            eval_params={"n_round":3, "eval_auc_dict":{}}, 
            pred_params={"do_estimators":False}
        ):
        """
        交差検証を行い、モデルを作成する.
        ※Fit関数と共通化したかったが、共通化するとpre_proc_を重複して計算するなど弊害が多く、分けて管理する
        Params::
            df_train: data frame
            save_traindata: train の推論結果を保存するかどうか
            conv_autostop:
                交差検証の中でauto stop が行われている場合は、その値を保持して学習時のmodelに渡す
                ["best_iteration_", "n_estimators"] : [交差検証時に拾ってくるparameterの値, その値を学習時に渡すパラメータ]
            split_params:
                split するための parameter. 詳しくはsplit_data_balance を参照
            fit_params:
                fit 時の追加のparameter を渡す事ができる. 特殊文字列として_validation_x, _validation_y があり、
                これは交差検証時のvalidation dataをその文字列と変換して渡す事ができる
                {"eval_set":("_validation_x", "_validation_y"), "early_stopping_rounds":50}
            fit_params_train:
                本番訓練時のパラメータ
            eval_params:
                評価時のparameter. {"n_round":3, "eval_auc_list":{}}
                eval_classification_model など参照
            pred_params:
                predict_detail など参照
        """
        self.logger.info("START")
        self.logger.info("df shape:%s, save_traindata:%s, split_params:%s, "+\
                         "fit_params:%s, eval_params:%s, pred_params:%s.", \
                         df_train.shape, save_traindata, \
                         split_params, fit_params, eval_params, pred_params)
        self.logger.info("features:%s", self.colname_explain.shape)
        if self.model is None:
            self.logger.raise_error("model is not set.")

        split_params = split_params.copy() if split_params is not None else {}
        fit_params   = fit_params.  copy() if fit_params   is not None else {}
        eval_params  = eval_params. copy() if eval_params  is not None else {}
        pred_params  = pred_params. copy() if pred_params  is not None else {}
        split_params["random_seed"] = self.random_seed

        # numpyに変換(前処理の結果を反映する
        X_train, Y_train = self.ndf_apply_preproc(df_train)

        ## データのスプリット
        split_params["random_seed"] = self.random_seed
        train_indexes, test_indexes = split_data_balance(Y_train[0], **split_params)

        # 交差検証開始
        i_split = 1
        n_epoch = 0 # autostopしていた場合、交差検証での最大値を引き継がせる
        df_score = pd.DataFrame()
        for train_index, test_index in zip(train_indexes, test_indexes):
            self.logger.info(f"create model by split samples, Cross Validation : {i_split} start...")
            _X_train = [_X[train_index] for _X in X_train]
            _Y_train = [_Y[train_index] for _Y in Y_train]

            # fit_params で _validation_x, _validation_y が記載された場合はValidation での 検証分の値を渡す
            ## 値を上書きするので、初回時しか_validation_xの文字列が有効にならないため、オリジナルのパラメータで比較する
            _fit_params = fit_params.copy()
            if _fit_params is not None and len(_fit_params) > 0:
                ## 追加の個別定義targetがある場合はtupleで渡してやる. なければ元に戻す
                _validation_x = [_X[test_index] for _X in X_train]
                _validation_y = [_Y[test_index] for _Y in Y_train]
                _validation_x = _validation_x[0] if len(_validation_x) == 1 else _validation_x
                _validation_y = _validation_y[0] if len(_validation_y) == 1 else _validation_y
                _fit_params = conv_validdata_in_fitparmas(_fit_params.copy(), _validation_x, _validation_y)

            self.model.fit((X_train[0][train_index] if len(X_train) == 1 else tuple([_X[train_index] for _X in X_train])), \
                           (Y_train[0][train_index] if len(Y_train) == 1 else tuple([_Y[train_index] for _Y in Y_train])), \
                           **_fit_params)
            self.logger.info("create model by split samples, Cross Validation : %s end...", i_split)
            # 結果の格納(訓練データと検証データの結果を格納する)
            for _type, _index in zip(["train", "test"], [train_index, test_index]):
                if save_traindata == False and _type == "train": continue
                self.logger.debug("predict %s dataset.", _type)

                # 訓練データの精度を記録
                pred_params["n_jobs"] = self.n_jobs
                dfwk = predict_detail(self.model,
                                       (X_train[0][_index] if len(X_train) == 1 else tuple([_X[_index] for _X in X_train])),
                                       **pred_params)
                dfwk["answer"]  = (Y_train[0][_index] if len(Y_train) == 1 else [_Y[_index] for _Y in Y_train])
                dfwk["index"]   = _index
                dfwk["type"]    = _type
                dfwk["i_split"] = i_split
                df_score = pd.concat([df_score, dfwk], axis=0, ignore_index=True, sort=False)

            # autostop が行われている場合は、各交差検証の中での最大値を保存する
            if conv_autostop is not None:
                _n_epoch = int(self.model.__getattribute__(conv_autostop[0]))
                n_epoch  = n_epoch if n_epoch > _n_epoch else _n_epoch
                
            i_split += 1
        self.index_valid = df_train.index[df_score["index"].values] # DaaFrame のインデックスを残しておく
        self.df_pred_valid  = df_score.copy()

        ## 評価する
        n_splits = i_split
        for _type in ["train","test"]:
            if save_traindata == False and _type == "train": continue
            df_eval = pd.DataFrame()
            for i_split in range(1, n_splits): # ここ前までn_splits+1 としていたが、上のforﾙｰﾌﾟで5回ったらi_split=6になるので不用に多かった...
                dfwkwk = self.df_pred_valid.copy()
                dfwkwk = dfwkwk[(dfwkwk["i_split"] == i_split) & (dfwkwk["type"]==_type)].copy()
                _, sewk = self.eval_model(dfwkwk, **eval_params)
                if df_eval.index.values.shape[0] == 0: df_eval = pd.DataFrame(index=sewk.index) # 初回のみインデックスをつけて初期化
                df_eval["i_split_"+str(i_split)] = sewk
                self.logger.debug("\n%s", df_eval) # DEBUG
            df_eval["mean"] = df_eval.loc[:, df_eval.columns.str.contains("^i_split_", regex=True)].mean(axis=1)
            df_eval["std"]  = df_eval.loc[:, df_eval.columns.str.contains("^i_split_", regex=True)].std(axis=1)

            ## 混合行列は全体で作成する
            dfwkwk  = self.df_pred_valid.copy()
            dfwkwk  = dfwkwk[(dfwkwk["type"]==_type)].copy()
            dfwk, _ = self.eval_model(dfwkwk, **eval_params)
            if _type == "test":
                self.logger.info("evaluation model by validation data.")
                self.eval_valid_cm  = dfwk.copy()
                self.eval_valid_val = (df_eval["mean"].astype(str) + " +/- " + df_eval["std"].astype(str)).copy()
                self.logger.info("\n%s", self.eval_valid_cm)
                self.logger.info("\n%s", self.eval_valid_val)

        ## 最後に全データでモデルを作成しておく
        split_params["n_splits"] = 1
        X_train, Y_train = None, None

        # early stopping を採用している場合、交差検証の中で最大のiteration回数を全体学習に採用する
        if conv_autostop is not None:
            # model の epoch回数を持っている要素に上書きする
            _epoch = 0
            _epoch = int(n_epoch) #常に最大値を採用する
            if _epoch < 10: _epoch = 10 # 最低10回は保証する
            self.model.__setattr__(conv_autostop[1], _epoch)

        # 全体でfitting
        logger.info("model training.")
        logger.info(f"{self.model}")
        self.fit(df_train, split_params=split_params, fit_params=fit_params_train, eval_params=eval_params, pred_params=pred_params)

        self.logger.info("END")


    # CalibratedClassifierCVは交差検証時にValidaionデータでfittingを行う
    # 本クラスでは独自に交差検証を実装しているため、交差するのが面倒くさい
    # なので、入力X(predict_proba)に対して、そのままpredict_probaが帰ってくるような
    # 擬似機械学習アルゴリズムを自作する
    class MyCalibrater:
        class _MockCalibrater:
            def __init__(self, classes):
                self.classes_ = classes
            def predict_proba(self, X):
                return X
            def __str__(self):
                return "MockCalibrater"

        def __init__(self, model):
            self.model    = model
            self.classes_ = self.model.classes_
            self.mock_calibrater = self._MockCalibrater(self.model.classes_)
            self.calibrater      = CalibratedClassifierCV(self.mock_calibrater, cv="prefit", method='isotonic')

        def __str__(self):
            return str(self.calibrater)
            

        # ここで入力するXはpredict_proba である
        # 実際の特徴量ではない点注意
        def fit(self, X, Y):
            self.calibrater.fit(X, Y)

        # ここで入力するXはpredict_proba である
        # 実際の特徴量ではない点注意
        def predict_proba_mock(self, X):
            return self.calibrater.predict_proba(X)

        # ここで入力するXは実際の特徴量である
        def predict_proba(self, X):
            return self.calibrater.predict_proba(self.model.predict_proba(X))
            
        # ここで入力するXは実際の特徴量である
        def predict(self, X):
            return self.calibrater.predict(self.model.predict_proba(X))


    def calibration(self, df: pd.DataFrame=None):
        """
        予測確率のキャリブレーション
        予測モデルは作成済みで別データでfittingする場合
        """
        self.logger.info("START")
        if self.is_model_fit == False:
            self.logger.raise_error("model is not fitted.")
        if self.is_classification_model() == False:
            self.logger.raise_error("model type is not classification")

        pred_prob_bef, pred_prob_aft = None, None
        if df is not None:
            # 正攻法のアルゴリズム
            ## キャリブレーション用モデルのインスタンス
            self.calibrater = CalibratedClassifierCV(self.model, cv="prefit", method='isotonic')
            self.logger.info("\n%s", self.calibrater)
            ## fittingを行う
            X = df[self.colname_explain].astype(np.float32).copy().values
            Y = df[self.colname_answer ].astype(np.float32).copy().values
            self.calibrater.fit(X, Y)

            pred_prob_bef = self.model.predict_proba(X)
            pred_prob_aft = self.calibrater.predict_proba(X)
        else:
            # 少し偏屈なアルゴリズム
            ## 擬似機械学習モデルのインスタンス
            self.calibrater = self.MyCalibrater(self.model)
            self.logger.info("\n%s", self.calibrater)
            ## fittingを行う
            dfwk = self.df_pred_valid.copy()
            if dfwk.shape[0] == 0:
                ## クロスバリデーションを行っていない場合はエラー
                self.logger.raise_error("cross validation is not done !!")

            X = dfwk.loc[:, dfwk.columns.str.contains("^predict_proba_")].values
            Y = dfwk["answer"].values
            self.calibrater.fit(X, Y)
            self.is_calibration = True # このフラグでON/OFFする

            # ここでのXは確率なので、mockを使って補正後の値を確認する
            pred_prob_bef = X
            pred_prob_aft = self.calibrater.predict_proba_mock(X)

        self.is_calibration = True # このフラグでON/OFFする

        # Calibration Curve Plot
        classes = np.sort(np.unique(Y).astype(int))
        self.fig["calibration_curve"] = plt.figure(figsize=(12, 8))
        ax1 = self.fig["calibration_curve"].add_subplot(2,1,1)
        ax2 = self.fig["calibration_curve"].add_subplot(2,1,2)
        ## ラベル数に応じて処理を分ける
        for i, i_class in enumerate(classes):
            ## 変化の前後を記述する
            fraction_of_positives, mean_predicted_value = calibration_curve((Y==i_class), pred_prob_bef[:, i], n_bins=10)
            ax1.plot(mean_predicted_value, fraction_of_positives, "s:", label="before_label_"+str(i_class))
            ax2.hist(pred_prob_bef[:, i], range=(0, 1), bins=10, label="before_label_"+str(i_class), histtype="step", lw=2)
            fraction_of_positives, mean_predicted_value = calibration_curve((Y==i_class), pred_prob_aft[:, i], n_bins=10)
            ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label="after_label_"+str(i_class))
        ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        ax1.legend()
        ax2.legend()
        ax1.set_title('Calibration plots  (reliability curve)')
        ax2.set_xlabel('Mean predicted value')
        ax1.set_ylabel('Fraction of positives')
        ax2.set_ylabel('Count')

        self.logger.info("END")


    def predict(self, df: pd.DataFrame=None, _X: np.ndarray=None, _Y: np.ndarray=None, pred_params={"do_estimators":False}):
        """
        モデルの予測
        Params::
            df: input DataFrame
            _X: ndarray (こちらが入力されたらpre_proc_などの処理が短縮される)
            _Y: ndarray (こちらが入力されたらpre_proc_などの処理が短縮される)
        """
        self.logger.info("START")
        self.logger.info("df:%s, X:%s, Y:%s, pred_params:%s", \
                         df if df is None else df.shape, \
                         _X if _X is None else _X.shape, \
                         _Y if _Y is None else _Y.shape, \
                         pred_params)
        if self.is_model_fit == False:
            self.logger.raise_error("model is not fitted.")

        # 下記の引数パターンの場合はエラーとする
        if (type(df) == type(None)) and (type(_X) == type(None)):
            self.logger.raise_error("df and _X is null.")

        pred_params  = pred_params.copy()
        pred_params["n_jobs"] = self.n_jobs
            
        # 前処理の結果を反映する

        X = None
        if _X is None:
            X, _ = self.ndf_apply_preproc(df, x_proc=True, y_proc=False)
            if len(X) == 1: X = X[0] #_addがなければ元に戻す
        else:
            # numpy変換処理での遅延を避けるため、引数でも指定できるようにする
            X = _X

        # 予測処理
        ## キャリブレーターの有無で渡すモデルを変える
        df_score = None
        if self.is_calibration: df_score = predict_detail(self.calibrater, X, **pred_params)
        else:                   df_score = predict_detail(self.model,      X, **pred_params)

        if df is not None:
            df_score["index"] = df.index.values.copy()

        try:
            Y = None
            if _Y is None:
                _, Y = self.ndf_apply_preproc(df, x_proc=False, y_proc=True)
                if len(Y) == 1: Y = Y[0] #_addがなければ元に戻す
            else:
                # numpy変換処理での遅延を避けるため、引数でも指定できるようにする
                Y = _Y
            df_score["answer"]  = Y
        except KeyError:
            self.logger.warning("answer is none. predict only.")
        except ValueError:
            # いずれは削除する
            self.logger.warning("Preprocessing answer's label is not work.")

        self.logger.info("END")
        return df_score


    # 偏差値に変換して予測する
    def predict_deviation_value(self, df=None, _X=None, _Y=None, pred_params={"do_estimators":False}):
        self.logger.info("START")
        # 通常の予測関数で予測結果を得る
        df_score = self.predict(df=df, _X=_X, _Y=_Y, pred_params=pred_params)

        # 交差検証を前提とする(また、予測結果がガウス分布になっている事を仮定する)
        if self.df_pred_valid.shape[0] > 0:
            dfwk = self.df_pred_valid
            df_score["predict"] = 10*(df_score["predict"] - dfwk["predict"].mean())/dfwk["predict"].std() + 50
        else:
            self.logger.raise_error("predict_cross_validation is not exist.")
        self.logger.info("END")
        return df_score


    # テストデータでの検証. 結果は上位に返却
    def predict_testdata(
            self, df_test, store_eval=False, eval_params={"n_round":3, "eval_auc_list":[]},
            pred_params={"do_estimators":False}
        ):
        """
        テストデータを予測し、評価する
        """
        self.logger.info("START")
        self.logger.info("df:%s, store_eval:%s, eval_params:%s, pred_params:%s", \
                         df_test.shape, store_eval, eval_params, pred_params)
        if self.is_model_fit == False:
            self.logger.raise_error("model is not fitted.")

        eval_params  = eval_params.copy()
        pred_params  = pred_params.copy()

        Y_test = df_test[self.colname_answer].astype(np.float32).values.copy()

        # 予測する
        df_score = self.predict(df_test, pred_params=pred_params)
        df_score["i_split"] = 0
        for x in self.colname_other: df_score["other_"+x] = df_test[x].values
        self.df_pred_test = df_score.copy()

        ## 評価値を格納
        df_conf, se_eval = self.eval_model(df_score, **eval_params)
        self.logger.info("\n%s", df_conf)
        self.logger.info("\n%s", se_eval)

        if store_eval == True:
            ## データを格納する場合(下手に上書きさせないためデフォルトはFalse)
            self.eval_test_cm  = df_conf.copy()
            self.eval_test_val = se_eval.copy()
            ## テストデータの割合を格納
            if self.is_classification_model():
                sewk = pd.DataFrame(Y_test.astype(int)).groupby(0).size().sort_index()
                self.n_tested_samples = {int(x):sewk[int(x)] for x in self.model.classes_}
            else:
                self.n_tested_samples = Y_test.shape[0]

        # ROC Curve plot(クラス分類時のみ行う)
        ## ラベル数分作成する
        if self.is_classification_model():
            for _x in df_score.columns[df_score.columns.str.contains("^predict_proba_")]:
                y_ans = df_score["answer"].values
                y_pre = df_score[_x].values
                lavel = int(_x.split("_")[-1])
                self.plot_roc_curve("roc_curve_"+str(lavel), (y_ans==lavel), y_pre)

        self.logger.info("END")
        return df_conf, se_eval


    # 各特長量をランダムに変更して重要度を見積もる
    ## ※個別定義targetに関してはランダムに変更することを想定しない!!
    def calc_permutation_importance(self, df, n_trial, calc_size=0.1, eval_method="roc_auc", eval_params={"pos_label":(1,0,), "eval_auc_list":[]}):
        self.logger.info("START")
        self.logger.info("df:%s, n_trial:%s, calc_size:%s, eval_method:%s, eval_params:%s", \
                         df.shape, n_trial, calc_size, eval_method, eval_params)
        self.logger.info("features:%s", self.colname_explain.shape)
        if self.model is None:
            self.logger.raise_error("model is not set.")

        # aucの場合は確立が出せるモデルであることを前提とする
        if (eval_method == "roc_auc") or (eval_method == "roc_auc_multi"):
            if is_callable(self.model, "predict_proba") == False:
                self.logger.raise_error("this model do not predict probability.")

        # 個別に追加定義したtargetがあれば作成する
        X, Y = self.ndf_apply_preproc(df)

        # まずは、正常な状態での評価値を求める
        ## 時間短縮のためcalc_sizeを設け、1未満のサイズではその割合のデータだけ
        ## n_trial の回数まわしてscoreを計算し、2群間のt検定をとる
        score_normal_list = np.array([])
        X_index_list = []
        if calc_size >= 1:
            df_score     = self.predict(_X=(X[0] if len(X)==1 else tuple(X)), \
                                        _Y=(Y[0] if len(Y)==1 else tuple(Y)), \
                                        pred_params={"do_estimators":False})
            score_normal = evalate(eval_method, df_score["answer"].values, df_score["predict"].values, \
                                df_score.loc[:, df_score.columns.str.contains("^predict_proba_")].values, \
                                **eval_params)
            score_normal_list = np.append(score_normal_list, score_normal)
            for i in range(n_trial):
                X_index_list.append(np.arange(X[0].shape[0]))
        else:
            # calc_sizeの割合に縮小したデータに対してn_trial回数回す
            for i in range(n_trial):
                _random_index = np.random.permutation(np.arange(X[0].shape[0]))[:int(X[0].shape[0]*calc_size)]
                X_index_list.append(_random_index.copy())
                df_score      = self.predict(_X=(X[0][_random_index] if len(X) == 1 else tuple([__X[_random_index] for __X in X])),
                                             _Y=(Y[0][_random_index] if len(Y) == 1 else tuple([__Y[_random_index] for __Y in Y])),
                                             pred_params={"do_estimators":False})
                score_normal  = evalate(eval_method, df_score["answer"].values, df_score["predict"].values, \
                                     df_score.loc[:, df_score.columns.str.contains("^predict_proba_")].values, \
                                     **eval_params)
                score_normal_list = np.append(score_normal_list, score_normal)
        self.logger.info(f"model normal score is {eval_method}={score_normal_list.mean()} +/- {score_normal_list.std()}")

        # 特徴量をランダムに変化させて重要度を計算していく
        self.feature_importances_modeling = pd.DataFrame(columns=["feature_name", "p_value", "t_value", \
                                                                  "score", "score_diff", "score_std"])
        ## 短縮のため、決定木モデルでimportanceが0の特徴量は事前に省く
        colname_except_list = np.array([])
        if self.feature_importances.shape[0] > 0:
            colname_except_list = self.feature_importances[self.feature_importances["importance"] == 0]["feature_name"].values.copy()
        for i_colname, colname in enumerate(self.colname_explain):
            index_colname = df[self.colname_explain].columns.get_indexer([colname]).min()
            self.logger.debug("step : %s, feature is shuffled : %s", i_colname, colname)
            if np.isin(colname, colname_except_list).min() == False:
                ## 短縮リストにcolnameが存在しない場合
                score_random_list = np.empty(0)
                X_colname_bk      = X[0][:, index_colname].copy() # 後で戻せるようにバックアップする
                ## ランダムに混ぜて予測させる
                for i in range(n_trial):
                    X[:, index_colname] = np.random.permutation(X_colname_bk).copy()
                    df_score = self.predict(_X=(X[0][X_index_list[i]] if len(X) == 1 else tuple([__X[X_index_list[i]] for __X in X])), \
                                            _Y=(Y[0][X_index_list[i]] if len(Y) == 1 else tuple([__Y[X_index_list[i]] for __Y in Y])), \
                                            pred_params={"do_estimators":False})
                    score    = evalate(eval_method, df_score["answer"].values, df_score["predict"].values, \
                                    df_score.loc[:, df_score.columns.str.contains("^predict_proba_")].values, \
                                    **eval_params)
                    score_random_list = np.append(score_random_list, score)

                _t, _p = np.nan, np.nan
                if calc_size >= 1:
                    # t検定により今回のスコアの分布を評価する
                    _t, _p = stats.ttest_1samp(score_random_list, score_random_list[-1])
                else:
                    # t検定(非等分散と仮定)により、ベストスコアと今回のスコアの分布を評価する
                    _t, _p = stats.ttest_ind(score_random_list, score_random_list, axis=0, equal_var=False, nan_policy='propagate')
                
                # 結果の比較(スコアは小さいほど良い)
                self.logger.info("random score: %s = %s +/- %s. p value = %s, statistic(t value) = %s, score_list:%s", \
                                 eval_method, score_random_list.mean(), score_random_list.std(), \
                                 _p, _t, score_random_list)

                # 結果の格納
                self.feature_importances_modeling = \
                    self.feature_importances_modeling.append({"feature_name":colname, "p_value":_p, "t_value":_t, \
                                                              "score":score_random_list.mean(), \
                                                              "score_diff":score_normal - score_random_list.mean(), \
                                                              "score_std":score_random_list.std()}, ignore_index=True)
                # ランダムを元に戻す
                X[0][:, index_colname] = X_colname_bk.copy()
            else:
                ## 短縮する場合
                self.logger.info("random score: omitted")
                # 結果の格納
                self.feature_importances_modeling = \
                    self.feature_importances_modeling.append({"feature_name":colname, "p_value":np.nan, "t_value":np.nan, \
                                                              "score":np.nan, "score_diff":np.nan, "score_std":np.nan}, ignore_index=True)
        self.logger.info("END")


    # Oputuna で少数なデータから(あまり時間をかけずに)ハイパーパラメータを探索する
    # さらに、既知のアルゴリズムに対する設定を予め行っておく。
    # ※色々とパラメータ固定してからoptunaさせる方がよい. 学習率とか..
    def search_hyper_params(
        self, df: pd.DataFrame, tuning_eval: str, n_trials: int=10, df_test: pd.DataFrame=None, dict_param="auto", iters: int=None,
        split_params: dict={"n_splits":1,"y_type":"cls","weight":"balance","is_bootstrap":False}, 
        fit_params: dict={}, eval_params: dict={}
    ):
        self.logger.info("START")

        # numpyに変換
        X, Y = self.ndf_apply_preproc(df)
        ## 各データが１種類の場合は、元に戻す
        if len(X) == 1: X = X[0]
        if len(Y) == 1: Y = Y[0]
        X_test, Y_test = None, None
        if df_test is not None:
            X_test, Y_test = self.ndf_apply_preproc(df_test)
            if len(X_test) == 1: X_test = X_test[0]
            if len(Y_test) == 1: Y_test = Y_test[0]

        # データの数を変えながら探索する
        self.optuna: optuna.study.Study = optuna.create_study()
        df_optuna = search_hyperparams_by_optuna(
            self.optuna, self.model, X, Y, n_trials=n_trials, iters=iters, X_test=X_test, Y_test=Y_test, dict_param=dict_param, 
            tuning_eval=tuning_eval, split_params=split_params, fit_params=fit_params, eval_params=eval_params, n_jobs=self.n_jobs
        )
        self.optuna_result = df_optuna.copy()
        self.logger.info("END")


    def plot_roc_curve(self, name, y_ans, y_pred_prob):
        self.logger.info("START")
        # canvas の追加
        self.fig[name] = plt.figure(figsize=(12, 8))
        ax = self.fig[name].add_subplot(1,1,1)

        fpr, tpr, _ = roc_curve(y_ans, y_pred_prob)
        _auc = auc(fpr, tpr)

        # ROC曲線をプロット
        ax.plot(fpr, tpr, label='ROC curve (area = %.3f)'%_auc)
        ax.legend()
        ax.set_title('ROC curve')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.grid(True)
        self.logger.info("END")


    def save(self, dir_path: str="./", mode: int=0, exist_ok=False, remake=False):
        """
        mymodel のデータを保存します
        Params::
            dir_path: 保存するpath
            mode:
                0 : 全て保存
                1 : 全体のpickleだけ保存
                2 : 全体のpickle以外を保存
        """
        self.logger.info("START")
        dir_path = correct_dirpath(dir_path)
        makedirs(dir_path, exist_ok=exist_ok, remake=remake)
        # 全データの保存
        if mode in [0,1]:
            save_pickle(self, dir_path + self.name + ".pickle")
        if mode in [0,2]:
            # モデルを保存
            if is_callable(self.model, "dump") == True:
                ## NNだとpickleで保存すると激重いので、dump関数がればそちらを優先する
                self.model.dump(dir_path + self.name + ".model.pickle")
            else:
                save_pickle(self.model, dir_path + self.name + ".model.pickle")
            # モデル情報をテキストで保存
            with open(dir_path + self.name + ".metadata", mode='w') as f:
                f.write("colname_explain_first="+str(self.colname_explain_first.tolist())+"\n")
                f.write("colname_explain="      +str(self.colname_explain.tolist())+"\n")
                f.write("colname_answer='"      +self.colname_answer+"'\n")
                f.write("n_trained_samples="    +str(self.n_trained_samples)+"\n")
                f.write("n_tested_samples="     +str(self.n_tested_samples)+"\n")
                for x in self.eval_train_val.index: f.write("train_"+x+"="+str(self.eval_train_val[x])+"\n")
                for x in self.eval_valid_val.index: f.write("validation_"+x+"="+str(self.eval_valid_val[x])+"\n")
                for x in self.eval_test_val. index: f.write("test_"+x+"="+str(self.eval_test_val [x])+"\n")
            # ログを保存
            with open(dir_path + self.name + ".log", mode='w') as f:
                f.write(self.logger.internal_stream.getvalue())
            # 画像を保存
            for _x in self.fig.keys():
                self.fig[_x].savefig(dir_path + self.name + "_" + _x + '.png')
            # CSVを保存
            self.feature_importances_randomtrees.to_csv(dir_path + self.name + ".feature_importances_randomtrees.csv", encoding="shift-jis")
            self.feature_importances_modeling   .to_csv(dir_path + self.name + ".feature_importances_modeling.csv",    encoding="shift-jis")
            self.feature_importances            .to_csv(dir_path + self.name + ".feature_importances.csv",             encoding="shift-jis")
            self.df_pred_train.to_pickle(dir_path + self.name + ".predict_train.pickle")
            self.df_pred_valid.to_pickle(dir_path + self.name + ".predict_valid.pickle")
            self.df_pred_test. to_pickle(dir_path + self.name + ".predict_test.pickle")
            self.eval_train_cm.to_csv(dir_path + self.name + ".eval_train_confusion_matrix.csv", encoding="shift-jis")
            self.eval_valid_cm.to_csv(dir_path + self.name + ".eval_valid_confusion_matrix.csv", encoding="shift-jis")
            self.eval_test_cm.to_csv( dir_path + self.name + ".eval_test_confusion_matrix.csv", encoding="shift-jis")
        self.logger.info("END")


    def load_colname_explain(self, filepath: str, mode: int=0):
        """
        特徴量リストを外部ファイルから読み込む
        Params::
            mode:
            0 : colname_explain をそのままコピー
            1 : feature_importances があれば重要度0は省く
            2 : 追加の特徴量がある事を考慮して, 読み込む対象のmymodelのcolname_explain_firstと比較して追加する
            3 : 読み込む対象のcolname_explainにあって、現モデルのcolname_explainにないカラムは省く
        """
        self.logger.info("START")
        self.logger.info(f"load other My model: {filepath}")
        mymodel = load_my_model(filepath)
        if   mode == 0:
            self.colname_explain = mymodel.colname_explain.copy()
        elif mode == 1:
            self.colname_explain = mymodel.colname_explain.copy()
            df = mymodel.feature_importances.copy()
            if df.shape[0] > 0:
                # 除外リストを作成し、除外対象以外の特徴量を残す
                colname_list = df[(df["importance"].isna()) | (df["importance"] == 0)]["feature_name"].values
                self.colname_explain = self.colname_explain[~np.isin(self.colname_explain, colname_list)]
        elif mode == 2:
            now_features  = self.colname_explain.copy()
            base_features = mymodel.colname_explain_first.copy()
            # 今のモデルの特徴量にあって、基本モデルの初期特徴量にないものを抽出
            add_features  = now_features[~np.isin(now_features, base_features)]
            self.colname_explain = np.append(mymodel.colname_explain.copy(), add_features.copy())
        elif mode == 3:
            now_features  = self.colname_explain_first.copy()
            base_features = mymodel.colname_explain.copy()
            self.colname_explain = base_features[np.isin(base_features, now_features)].copy()

        self.logger.info("END")


    def load_model(self, filepath: str, mode: int=0):
        """
        モデルだけを別ファイルから読み込む
        Params::
            mode:
                0: model をそのままコピー
                1: optuna のbest params をロード
        """
        self.logger.info("START")
        self.logger.info(f"load other My model :{filepath}")
        mymodel = load_my_model(filepath)
        if   mode == 0:
            self.model = mymodel.model.copy()
        elif mode == 1:
            best_params = mymodel.optuna_study.best_params.copy()
            ## 現行モデルに上書き
            self.model = self.model.set_params(**best_params)
        self.logger.info(f"\n{self.model}", )
        self.logger.info("END")


def load_my_model(filepath: str) -> MyModel:
    """
    MyModel形式をloadする
    Params::
        filepath: 全部が保存されたpickle名か、model名までのpathを指定する
    """
    logger.info("START")
    mymodel = None
    if filepath.rfind(".pickle") == len(filepath) - len(".pickle"):
        # ファイル名の末尾が.pickleの場合
        mymodel = load_pickle(filepath)
        mymodel.__class__ = MyModel
        mymodel.logger = set_logger(_logname + "." + mymodel.name, log_level="info", internal_log=True)
    else:
        logger.raise_error(f"filepath: {filepath}. we can load only 'mymodel pickle' file")
    logger.info("END")
    return mymodel

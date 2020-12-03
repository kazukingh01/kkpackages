import datetime
from typing import List, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import optuna
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# local package
from kkpackage.lib.learning import ProcRegistry, Calibrater, MyReplaceValue, MyAsType, MyReshape, MyDropNa, MyMinMaxScaler, MyFillNaMinMax
from kkpackage.util.learning import search_features_by_variance, search_features_by_correlation, \
    split_data_balance, predict_detail, evalate, eval_classification_model, eval_regressor_model, \
    is_classification_model, conv_validdata_in_fitparmas, calc_randomtree_importance, calc_parallel_mutual_information
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
        colname_explain: np.ndarray, colname_answer: np.ndarray, colname_other: np.ndarray = None,
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
        self.colname_explain       = conv_ndarray(colname_explain)
        self.colname_explain_hist  = []
        self.colname_answer        = conv_ndarray([colname_answer]) if isinstance(colname_answer, str) else conv_ndarray(colname_answer)
        self.colname_other         = conv_ndarray(colname_other) if colname_other is not None else np.array([])
        self.df_correlation                     = pd.DataFrame()
        self.df_feature_importances             = pd.DataFrame()
        self.df_feature_importances_randomtrees = pd.DataFrame()
        self.df_feature_importances_modeling    = pd.DataFrame()
        self.df_adversarial_valid               = pd.DataFrame()
        self.df_adversarial_importances         = pd.DataFrame()
        self.model          = model
        self.is_model_fit   = False
        self.optuna         = None
        self.calibrater     = None
        self.is_calibration = False
        self.preproc        = ProcRegistry(self.colname_explain, self.colname_answer)
        self.n_trained_samples = {}
        self.n_tested_samples  = {}
        self.index_train = np.array([]) # 実際に残す際はマルチインデックスかもしれないので、numpy形式にはならない
        self.index_valid = np.array([]) # 実際に残す際はマルチインデックスかもしれないので、numpy形式にはならない
        self.index_test  = np.array([]) # 実際に残す際はマルチインデックスかもしれないので、numpy形式にはならない
        self.df_pred_train = pd.DataFrame()
        self.df_pred_valid = pd.DataFrame()
        self.df_pred_test  = pd.DataFrame()
        self.df_cm_train   = pd.DataFrame()
        self.df_cm_valid   = pd.DataFrame()
        self.df_cm_test    = pd.DataFrame()
        self.se_eval_train = pd.Series(dtype=object)
        self.se_eval_valid = pd.Series(dtype=object)
        self.se_eval_test  = pd.Series(dtype=object)
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
            **params: その他インスタンスに追加したい変数
        """
        self.logger.debug("START")
        self.model          = model
        self.optuna         = None
        self.calibrater     = None
        self.is_calibration = False
        self.is_model_fit   = False
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
    

    def update_features(self, cut_features: np.ndarray, alive_features: np.ndarray=None):
        self.logger.info("START")
        self.colname_explain_hist.append(self.colname_explain.copy())
        if alive_features is None:
            self.colname_explain = self.colname_explain[~np.isin(self.colname_explain, cut_features)]
        else:
            cut_features         = self.colname_explain[~np.isin(self.colname_explain, alive_features)]
            self.colname_explain = self.colname_explain[ np.isin(self.colname_explain, alive_features)]
        self.preproc.set_columns(self.colname_explain, type_proc="x")
        self.logger.info(f"cut   features :{cut_features.shape[0]        }. features...{cut_features}")
        self.logger.info(f"alive features :{self.colname_explain.shape[0]}. features...{self.colname_explain}")
        self.logger.info("END")


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
        cut_features = search_features_by_variance(df, cutoff=cutoff, ignore_nan=ignore_nan, n_jobs=self.n_jobs)
        self.update_features(cut_features) # 特徴量の更新
        self.logger.info("END")


    def cut_features_by_correlation(self, df, cutoff=0.9, ignore_nan_mode=0, n_div_col=1, on_gpu_size=1, min_n_not_nan=10):
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
        df = self.preproc(df, x_proc=False, y_proc=False, row_proc=True)
        df_corr, _ = search_features_by_correlation(
            df[self.colname_explain], cutoff=cutoff, ignore_nan_mode=ignore_nan_mode, 
            n_div_col=n_div_col, on_gpu_size=on_gpu_size, min_n_not_nan=min_n_not_nan, n_jobs=self.n_jobs
        )
        self.df_correlation = df_corr.copy()
        alive_features, cut_features = self.features_by_correlation(cutoff)
        self.update_features(cut_features, alive_features=alive_features) # 特徴量の更新
        self.logger.info("END")


    def cut_features_by_random_tree_importance(
        self, df: pd.DataFrame=None, cut_ratio: float=0, sort: bool=True, calc_randomtrees: bool=False, **kwargs
    ):
        self.logger.info("START")
        self.logger.info("cut_ratio:%s", cut_ratio)
        if self.model is None:
            self.logger.raise_error("model is not set !!")
        if calc_randomtrees:
            if df is None: self.logger.raise_error("dataframe is None !!")
            if self.colname_answer.shape[0] > 1: self.logger.raise_error(f"answer has over 1 columns. {self.colname_answer}")
            df   = self.preproc(df, x_proc=False, y_proc=False, row_proc=True)
            proc = ProcRegistry(self.colname_explain, self.colname_answer)
            proc.register(
                [
                    MyAsType(np.float32),
                    MyReplaceValue(float( "inf"), float("nan")), 
                    MyReplaceValue(float("-inf"), float("nan")),
                    MyFillNaMinMax(),
                ], type_proc="x"
            )
            proc.register(
                [
                    MyAsType(np.int32),
                ], type_proc="y"
            )
            proc.fit(df)
            X, Y = proc(df, autofix=True, x_proc=True, y_proc=True, row_proc=True)
            self.df_feature_importances_randomtrees = calc_randomtree_importance(
                X, Y, colname_explain=self.colname_explain, 
                is_cls_model=self.is_classification_model(), n_jobs=self.n_jobs, **kwargs
            )
        if sort:
            self.logger.info("sort features by randomtree importance.")
            self.colname_explain_hist.append(self.colname_explain.copy())
            self.colname_explain = self.df_feature_importances_randomtrees["feature_name"].values.copy()
        if cut_ratio > 0:
            self.logger.info(f"cut features by randomtree importance. cut_ratio: {cut_ratio}")
            alive_features, cut_features = self.features_by_random_tree_importance(cut_ratio)
            self.update_features(cut_features, alive_features=alive_features) # 特徴量の更新
        self.logger.info("END")
    

    def cut_features_by_mutual_information(self, df: pd.DataFrame, calc_size: int=50, bins: int=10, base_max: int=1):
        self.logger.info("START")
        df = self.preproc(df, x_proc=False, y_proc=False, row_proc=True)
        proc = ProcRegistry(self.colname_explain, self.colname_answer)
        proc.register(
            [
                MyAsType(np.float16),
                MyReplaceValue(float( "inf"), float("nan")), 
                MyReplaceValue(float("-inf"), float("nan")),
                MyMinMaxScaler(feature_range=(0, base_max - (1./bins/10.))),
            ], type_proc="x"
        )
        proc.fit(df)
        ndf_x, _ = proc(df, autofix=True, x_proc=True, y_proc=False, row_proc=False)
        df       = pd.DataFrame(ndf_x, columns=self.colname_explain)
        self.df_mutual_information = calc_parallel_mutual_information(df, n_jobs=self.n_jobs, calc_size=calc_size, bins=bins, base_max=base_max)
        self.logger.info("END")


    def features_by_correlation(self, cutoff: float) -> (np.ndarray, np.ndarray):
        self.logger.info("START")
        columns = self.df_correlation.columns.values.copy()
        alive_features = columns[(((self.df_correlation > cutoff) | (self.df_correlation < -1*cutoff)).sum(axis=0) == 0)].copy()
        cut_list       = columns[~np.isin(columns, alive_features)]
        self.logger.info("END")
        return alive_features, cut_list


    def features_by_adversarial_validation(self, cutoff: float) -> (np.ndarray, np.ndarray):
        self.logger.info("START")
        columns = self.df_adversarial_importances["feature_name"].values.copy()
        cut_list       = columns[(self.df_adversarial_importances["importance"] > cutoff).values]
        alive_features = columns[~np.isin(columns, cut_list)]
        self.logger.info("END")
        return alive_features, cut_list


    def features_by_random_tree_importance(self, cut_ratio: float) -> (np.ndarray, np.ndarray):
        self.logger.info("START")
        self.logger.info("cut_ratio:%s", cut_ratio)
        if self.model is None:
            self.logger.raise_error("model is not set !!")
        if self.df_feature_importances_randomtrees.shape[0] == 0:
            self.logger.raise_error("df_feature_importances_randomtrees is None. You should do calc_randomtree_importance() first !!")
        _n = int(self.df_feature_importances_randomtrees.shape[0] * cut_ratio)
        alive_features = self.df_feature_importances_randomtrees.iloc[:-1*_n ]["feature_name"].values.copy()
        cut_list       = self.df_feature_importances_randomtrees.iloc[ -1*_n:]["feature_name"].values.copy()
        self.logger.info("END")
        return alive_features, cut_list
    

    def set_default_proc(self, df: pd.DataFrame):
        self.preproc.register(
            [
                MyAsType(np.float32),
                MyReplaceValue(float( "inf"), float("nan")), 
                MyReplaceValue(float("-inf"), float("nan"))
            ], type_proc="x"
        )
        self.preproc.register(
            [
                (MyAsType(np.int32) if self.is_classification_model() else MyAsType(np.float32)),
                MyReshape(-1),
            ], type_proc="y"
        )
        self.preproc.register(
            [
                MyDropNa(self.colname_answer)
            ], type_proc="row"
        )
        self.preproc.fit(df)


    def eval_model(self, df_score, **eval_params) -> (pd.DataFrame, pd.Series, ):
        """
        回帰や分類を意識せずに評価値を出力する
        """
        df_conf, se_eval = pd.DataFrame(), pd.Series(dtype=object)
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

        # numpyに変換(前処理の結果を反映させる)
        X_train, Y_train = self.preproc(df_train, autofix=False)

        ## データのスプリット
        split_params["random_seed"] = self.random_seed
        if   len(Y_train[0].shape) == 1:
            train_indexes, _ = split_data_balance(Y_train[0], **split_params)
        elif len(Y_train[0].shape) == 2:
            train_indexes, _ = split_data_balance(Y_train[0][:, 0], **split_params)
        else:
            logger.raise_error(f'y_train shape is over: {Y_train[0].shape}')

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
            # np.bincount(Y_train.astype(int)) だと マイナスの値でエラーが出る
            sewk = pd.DataFrame(Y_train.astype(int)).groupby(0).size().sort_index()
            if is_callable(self.model, "classes_") == False:
                ## classes_ がない場合は手動で追加する
                self.model.classes_ = np.sort(np.unique(Y_train)).astype(int)
            self.n_trained_samples = {int(x):sewk[int(x)] for x in self.model.classes_}
        self.logger.debug("create model by All samples : end ...")

        # 訓練データの精度を記録
        pred_params["n_jobs"] = self.n_jobs
        df_score = predict_detail(self.model, X_train, **pred_params)
        if   type(Y_train) == tuple:
            pass
        elif type(Y_train) == np.ndarray and len(Y_train.shape) == 1: 
            df_score["answer"] = Y_train
        elif type(Y_train) == np.ndarray and len(Y_train.shape) == 2:
            for i in np.arange(Y_train.shape[1]):
                df_score["answer_"+str(i)] = Y_train[:, i]
        else:
            logger.warning(f'Y_train shape is over: {Y_train}')
        df_score["index"]  = train_indexes[0]
        for x in self.colname_other: df_score["other_"+x] = df_train.iloc[train_indexes[0], df_train.columns.get_indexer([x]).min()].copy().values
        self.index_train   = df_train.index[train_indexes[0]].copy() # DataFrame のインデックスを残しておく
        self.df_pred_train = df_score.copy()
        
        ## 評価する. NNのように複数の出力があるモデルは対象外とする
        if df_score.columns.isin(["predict", "answer"]).sum() == 2:
            df_conf, se_eval = self.eval_model(df_score, **eval_params)
            self.logger.info("evaluation model by train data.")
            self.df_cm_train  = df_conf.copy()
            self.se_eval_train = se_eval.astype(str).copy()
            self.logger.info(f'\n{self.df_cm_train}\n{self.se_eval_train}')

        # 特徴量の重要度
        self.logger.info("feature importance saving...")
        if is_callable(self.model, "feature_importances_") == True:
            _df = pd.DataFrame(np.array([self.colname_explain, self.model.feature_importances_]).T, columns=["feature_name","importance"])
            _df = _df.sort_values(by="importance", ascending=False).reset_index(drop=True)
            self.df_feature_importances = _df.copy()

        self.is_model_fit = True
        self.logger.info("END")


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
        self.logger.info(
            f'df shape: {df_train.shape}, save_traindata: {save_traindata}, conv_autostop: {conv_autostop}, split_params: {split_params}' +
            f'fit_params: {fit_params}, fit_params_train: {fit_params_train}, eval_params: {eval_params}, pred_params: {pred_params}'
        )
        self.logger.info(f"features: {self.colname_explain.shape}")
        if self.model is None:
            self.logger.raise_error("model is not set.")

        split_params = split_params.copy() if split_params is not None else {}
        fit_params   = fit_params.  copy() if fit_params   is not None else {}
        eval_params  = eval_params. copy() if eval_params  is not None else {}
        pred_params  = pred_params. copy() if pred_params  is not None else {}

        # numpyに変換(前処理の結果を反映する
        X_train, Y_train = self.preproc(df_train, autofix=False)

        ## データのスプリット
        split_params["random_seed"] = self.random_seed
        if   len(Y_train[0].shape) == 1:
            train_indexes, test_indexes = split_data_balance(Y_train[0], **split_params)
        elif len(Y_train[0].shape) == 2:
            train_indexes, test_indexes = split_data_balance(Y_train[0][:, 0], **split_params)
        else:
            logger.raise_error(f'y_train shape is over: {Y_train[0].shape}')

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
                dfwk = predict_detail(self.model, (X_train[0][_index] if len(X_train) == 1 else tuple([_X[_index] for _X in X_train])), **pred_params)
                if   len(Y_train) == 1 and type(Y_train[0]) == np.ndarray and len(Y_train[0].shape) == 1: 
                    dfwk["answer"] = Y_train[0][_index]
                elif len(Y_train) == 1 and type(Y_train[0]) == np.ndarray and len(Y_train[0].shape) == 2:
                    for _i in np.arange(Y_train[0].shape[1]):
                        dfwk["answer_"+str(_i)] = Y_train[0][_index, _i]
                else:
                    logger.warning(f'Y_train shape is over: {Y_train}')
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
        if df_score.columns.isin(["predict", "answer"]).sum() == 2:
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
                    self.df_cm_valid  = dfwk.copy()
                    self.se_eval_valid = (df_eval["mean"].astype(str) + " +/- " + df_eval["std"].astype(str)).copy()
                    self.logger.info("\n%s", self.df_cm_valid)
                    self.logger.info("\n%s", self.se_eval_valid)

        ## 最後に全データでモデルを作成しておく
        split_params["n_splits"] = 1
        del X_train, Y_train

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
    

    def adversarial_validation(
            self, df_train: pd.DataFrame, df_test: pd.DataFrame, use_answer: bool=False,
            model=None, n_splits: int=5, n_estimators: int=1000
        ):
        """
        adversarial validation. テストデータのラベルを判別するための交差顕彰
        testdataかどうかを判別しやすい特徴を省いたり、test dataの分布に近いデータを選択する事に使用する
        Params::
            df_train: train data
            df_test: test data
            model: 分類モデルのインスタンス
            n_splits: 何分割交差顕彰を行うか
        """
        self.logger.info("START")
        self.logger.info(f'df_train shape: {df_train.shape}, df_test shape: {df_test.shape}')
        self.logger.info(f"features: {self.colname_explain.shape}, answer: {self.colname_answer}")
        if model is None: model = ExtraTreesClassifier(n_estimators=n_estimators, n_jobs=self.n_jobs)
        self.logger.info(f'model: \n{model}')
        # numpyに変換(前処理の結果を反映する
        X_train, Y_train = self.preproc(df_train, autofix=True, y_proc=use_answer, x_proc=True, row_proc=True)
        X_test,  Y_test  = self.preproc(df_test,  autofix=True, y_proc=use_answer, x_proc=True, row_proc=True)
        if use_answer:
            # データをくっつける(正解ラベルも使う)
            X_train = np.concatenate([X_train, Y_train.reshape(-1).reshape(-1, 1)], axis=1).astype(X_train.dtype) #X_trainの型でCASTする
            X_test  = np.concatenate([X_test,  Y_test .reshape(-1).reshape(-1, 1)], axis=1).astype(X_test.dtype ) #X_test の型でCASTする
        Y_train = np.concatenate([np.zeros(X_train.shape[0]), np.ones(X_test.shape[0])], axis=0).astype(np.int32) # 先にこっちを作る
        X_train = np.concatenate([X_train, X_test], axis=0).astype(X_train.dtype) # 連結する
        ## データのスプリット.(under samplingにしておく)
        train_indexes, test_indexes = split_data_balance(Y_train, n_splits=n_splits, y_type="cls", weight="balance", is_bootstrap=False, random_seed=self.random_seed)

        # 交差検証開始
        i_split = 1
        df_score, df_importance = pd.DataFrame(), pd.DataFrame()
        for train_index, test_index in zip(train_indexes, test_indexes):
            self.logger.info(f"create model by split samples, Cross Validation : {i_split} start...")
            _X_train = X_train[train_index]
            _Y_train = Y_train[train_index]
            _X_test  = X_train[test_index]
            _Y_test  = Y_train[test_index]
            model.fit(_X_train, _Y_train)
            self.logger.info("create model by split samples, Cross Validation : %s end...", i_split)
            # 結果の格納
            dfwk = predict_detail(model, _X_test)
            dfwk["answer"] = _Y_test
            dfwk["index"]  = test_index # concat前のtrain dataのindexは意味ある
            dfwk["type"]    = "test"
            dfwk["i_split"] = i_split
            df_score = pd.concat([df_score, dfwk], axis=0, ignore_index=True, sort=False)
            i_split += 1
            # 特徴量の重要度
            if is_callable(model, "feature_importances_") == True:
                if use_answer:
                    _df = pd.DataFrame(np.array([self.colname_explain.tolist() + self.colname_answer.tolist(), model.feature_importances_]).T, columns=["feature_name","importance"])
                else:
                    _df = pd.DataFrame(np.array([self.colname_explain.tolist(), model.feature_importances_]).T, columns=["feature_name","importance"])
                _df = _df.sort_values(by="importance", ascending=False).reset_index(drop=True)
                df_importance = pd.concat([df_importance, _df.copy()], axis=0, ignore_index=True, sort=False)
        df_score["index_df"] = -1
        df_score.loc[(df_score["answer"] == 0), "index_df"] = df_train.index[df_score.loc[(df_score["answer"] == 0), "index"].values]
        df_score.loc[(df_score["answer"] == 1), "index_df"] = df_test .index[df_score.loc[(df_score["answer"] == 1), "index"].values - df_train.shape[0]]
        self.logger.info(f'{eval_classification_model(df_score, "answer", "predict", ["predict_proba_0", "predict_proba_1"])}')
        self.df_adversarial_valid = df_score.copy()
        if df_importance.shape[0] > 0:
            df_importance["importance"] = df_importance["importance"].astype(float)
            self.df_adversarial_importances = df_importance.groupby("feature_name")["importance"].mean().reset_index().sort_values("importance", ascending=False)
        self.logger.info("END")


    def calibration(self):
        """
        予測確率のキャリブレーション
        予測モデルは作成済みで別データでfittingする場合
        """
        self.logger.info("START")
        if self.is_model_fit == False:
            self.logger.raise_error("model is not fitted.")
        if self.is_classification_model() == False:
            self.logger.raise_error("model type is not classification")
        self.calibrater = Calibrater(self.model)
        self.logger.info("\n%s", self.calibrater)
        ## fittingを行う
        df = self.df_pred_valid.copy()
        if df.shape[0] == 0:
            ## クロスバリデーションを行っていない場合はエラー
            self.logger.raise_error("cross validation is not done !!")
        X = df.loc[:, df.columns.str.contains("^predict_proba_")].values
        Y = df["answer"].values
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


    def predict(self, df: pd.DataFrame=None, _X: np.ndarray=None, _Y: np.ndarray=None, pred_params={"do_estimators":False}, row_proc: bool=True):
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
        if df is not None and row_proc: df = self.preproc(df, x_proc=False, y_proc=False, row_proc=True) # ここで変換しないとdf_scoreとのindexが合わない
        if _X is None:
            X, _ = self.preproc(df, x_proc=True, y_proc=False, row_proc=False)
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
            for x in self.colname_other: df_score["other_"+x] = df[x].copy().values

        try:
            Y = None
            if _Y is None and df is not None:
                _, Y = self.preproc(df, x_proc=False, y_proc=True, row_proc=False)
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


    # テストデータでの検証. 結果は上位に返却
    def predict_testdata(
            self, df_test, store_eval=False, eval_params={"n_round":3, "eval_auc_list":[]},
            pred_params={"do_estimators":False}, row_proc: bool=True
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
        # 予測する
        df_score = self.predict(df=df_test, pred_params=pred_params, row_proc=row_proc)
        df_score["i_split"] = 0
        for x in self.colname_other: df_score["other_"+x] = df_test[x].values
        self.df_pred_test = df_score.copy()
        ## 評価値を格納
        df_conf, se_eval = self.eval_model(df_score, **eval_params)
        self.logger.info("\n%s", df_conf)
        self.logger.info("\n%s", se_eval)
        if store_eval == True:
            ## データを格納する場合(下手に上書きさせないためデフォルトはFalse)
            self.df_cm_test  = df_conf.copy()
            self.se_eval_test = se_eval.copy()
            ## テストデータの割合を格納
            if self.is_classification_model():
                sewk = pd.DataFrame(df_score["answer"].values.astype(int)).groupby(0).size().sort_index()
                self.n_tested_samples = {int(x):sewk[int(x)] for x in self.model.classes_}
            else:
                self.n_tested_samples = df_score["answer"].shape[0]
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
        X, Y = self.preproc(df)

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
        self.df_feature_importances_modeling = pd.DataFrame(columns=["feature_name", "p_value", "t_value", \
                                                                  "score", "score_diff", "score_std"])
        ## 短縮のため、決定木モデルでimportanceが0の特徴量は事前に省く
        colname_except_list = np.array([])
        if self.df_feature_importances.shape[0] > 0:
            colname_except_list = self.df_feature_importances[self.df_feature_importances["importance"] == 0]["feature_name"].values.copy()
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
                self.df_feature_importances_modeling = \
                    self.df_feature_importances_modeling.append({"feature_name":colname, "p_value":_p, "t_value":_t, \
                                                              "score":score_random_list.mean(), \
                                                              "score_diff":score_normal - score_random_list.mean(), \
                                                              "score_std":score_random_list.std()}, ignore_index=True)
                # ランダムを元に戻す
                X[0][:, index_colname] = X_colname_bk.copy()
            else:
                ## 短縮する場合
                self.logger.info("random score: omitted")
                # 結果の格納
                self.df_feature_importances_modeling = \
                    self.df_feature_importances_modeling.append({"feature_name":colname, "p_value":np.nan, "t_value":np.nan, \
                                                              "score":np.nan, "score_diff":np.nan, "score_std":np.nan}, ignore_index=True)
        self.logger.info("END")


    # Oputuna で少数なデータから(あまり時間をかけずに)ハイパーパラメータを探索する
    # さらに、既知のアルゴリズムに対する設定を予め行っておく。
    # ※色々とパラメータ固定してからoptunaさせる方がよい. 学習率とか..
    def search_hyper_params(
        self, df: pd.DataFrame, tuning_eval: str, n_trials: int=10, df_test: pd.DataFrame=None, dict_param="auto", iters: int=None,
        split_params: dict={"n_splits":1,"y_type":"cls","weight":"balance","is_bootstrap":False}, 
        fit_params: dict={}, eval_params: dict={}, storage: str=None
    ):
        """
        Params::
            storage: optuna の履歴を保存する
        """
        self.logger.info("START")
        # numpyに変換
        X, Y = self.preproc(df)
        ## 各データが１種類の場合は、元に戻す
        if len(X) == 1: X = X[0]
        if len(Y) == 1: Y = Y[0]
        X_test, Y_test = None, None
        if df_test is not None:
            X_test, Y_test = self.preproc(df_test, autofix=True, x_proc=True, y_proc=True, row_proc=True)
        # データの数を変えながら探索する
        if not storage:
            self.optuna: optuna.study.Study = optuna.create_study(
                study_name='optuna_'+datetime.datetime.now().strftime("%Y%m%d%H%M%S"),
                storage='sqlite:///optuna_'+datetime.datetime.now().strftime("%Y%m%d%H%M%S")+'.db',
            )
        else:
            self.optuna: optuna.study.Study = optuna.load_study(
                study_name='optuna_'+storage.split(".")[-2].split("_")[-1], storage=storage
            )
        df_optuna, best_params = search_hyperparams_by_optuna(
            self.optuna, self.model, X, Y, n_trials=n_trials, iters=iters, X_test=X_test, Y_test=Y_test, dict_param=dict_param, 
            tuning_eval=tuning_eval, split_params=split_params, fit_params=fit_params, eval_params=eval_params, n_jobs=self.n_jobs
        )
        self.optuna_result = df_optuna.copy()
        self.optuna_params = best_params
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
                3 : 最低限のデータのみをpickleだけ保存
        """
        self.logger.info("START")
        dir_path = correct_dirpath(dir_path)
        makedirs(dir_path, exist_ok=exist_ok, remake=remake)
        if mode in [0,2]:
            # モデルを保存
            if is_callable(self.model, "dump") == True:
                ## NNだとpickleで保存すると激重いので、dump関数がればそちらを優先する
                self.model.dump(dir_path + self.name + ".model.pickle")
            else:
                save_pickle(self.model, dir_path + self.name + ".model.pickle")
            # モデル情報をテキストで保存
            with open(dir_path + self.name + ".metadata", mode='w') as f:
                f.write("colname_explain_first="+str(self.colname_explain.tolist() if len(self.colname_explain_hist) == 0 else self.colname_explain_hist[0])+"\n")
                f.write("colname_explain="      +str(self.colname_explain.tolist())+"\n")
                f.write("colname_answer='"      +str(self.colname_answer.tolist())+"'\n")
                f.write("n_trained_samples="    +str(self.n_trained_samples)+"\n")
                f.write("n_tested_samples="     +str(self.n_tested_samples)+"\n")
                for x in self.se_eval_train.index: f.write("train_"+x+"="+str(self.se_eval_train[x])+"\n")
                for x in self.se_eval_valid.index: f.write("validation_"+x+"="+str(self.se_eval_valid[x])+"\n")
                for x in self.se_eval_test. index: f.write("test_"+x+"="+str(self.se_eval_test [x])+"\n")
            # ログを保存
            with open(dir_path + self.name + ".log", mode='w') as f:
                f.write(self.logger.internal_stream.getvalue())
            # 画像を保存
            for _x in self.fig.keys():
                self.fig[_x].savefig(dir_path + self.name + "_" + _x + '.png')
            # CSVを保存
            self.df_feature_importances_randomtrees.to_csv(dir_path + self.name + ".df_feature_importances_randomtrees.csv", encoding="shift-jis")
            self.df_feature_importances_modeling   .to_csv(dir_path + self.name + ".df_feature_importances_modeling.csv",    encoding="shift-jis")
            self.df_feature_importances            .to_csv(dir_path + self.name + ".df_feature_importances.csv",             encoding="shift-jis")
            self.df_pred_train.to_pickle(dir_path + self.name + ".predict_train.pickle")
            self.df_pred_valid.to_pickle(dir_path + self.name + ".predict_valid.pickle")
            self.df_pred_test. to_pickle(dir_path + self.name + ".predict_test.pickle")
            self.df_cm_train.to_csv(dir_path + self.name + ".eval_train_confusion_matrix.csv", encoding="shift-jis")
            self.df_cm_valid.to_csv(dir_path + self.name + ".eval_valid_confusion_matrix.csv", encoding="shift-jis")
            self.df_cm_test.to_csv( dir_path + self.name + ".eval_test_confusion_matrix.csv", encoding="shift-jis")
        # 全データの保存
        if mode in [0,1]:
            save_pickle(self, dir_path + self.name + ".pickle")
        if mode in [3]:
            ## 重いデータを削除する
            self.fig = {}
            self.df_correlation                     = pd.DataFrame()
            self.df_feature_importances             = pd.DataFrame()
            self.df_feature_importances_randomtrees = pd.DataFrame()
            self.df_feature_importances_modeling    = pd.DataFrame()
            self.df_adversarial_valid               = pd.DataFrame()
            self.df_adversarial_importances         = pd.DataFrame()
            self.df_pred_train = pd.DataFrame()
            self.df_pred_valid = pd.DataFrame()
            self.df_pred_test  = pd.DataFrame()
            self.df_cm_train   = pd.DataFrame()
            self.df_cm_valid   = pd.DataFrame()
            self.df_cm_test    = pd.DataFrame()
            save_pickle(self, dir_path + self.name + ".min.pickle")
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
            best_params = mymodel.optuna.best_params.copy()
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

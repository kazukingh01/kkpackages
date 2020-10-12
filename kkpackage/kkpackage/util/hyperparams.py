from functools import partial
import numpy as np
import pandas as pd
import optuna
# local package
from kkpackage.util.learning import evalate, split_data_balance, conv_validdata_in_fitparmas, is_classification_model
from kkpackage.util.common import is_callable
from kkpackage.util.logger import set_logger
logger = set_logger(__name__)

# optuna に埋め込むベース関数
# X, Y はそれぞれ１種類を想定する
def optuna_base_function(
    X: np.ndarray, Y: np.ndarray, X_test: np.ndarray, Y_test: np.ndarray, 
    model: object, dict_param: dict, tuning_eval: str, split_params: dict, fit_params: dict, eval_params: dict, 
    trial
):
    """
    optuna に埋め込むためのBaseFunction
    X: 訓練データ
    Y: 訓練正解ラベル
    X_test: テストデータ
    Y_test: テスト正解ラベル
    model: ハイパーパラメータサーチしたいモデル
    dict_param = {"learning_rate":["category",0.05,0.1,0.2,0.3,0.5], "n_estimators":["int",500,1500], "max_depth":["int",3,10], 
                  "gamma":["float",0.,0.3], "min_child_weight":["int",1,20], "subsample":["step", 0.5, 0.9, 0.1], 
                  "colsample_bytree":["step", 0.1, 0.9, 0.1], "random_state":["const",1], "n_jobs":["const", -1]}
    tuning_eval: kkpackage.util.features.evalate にあるtuning_eval
    split_params:
        この設定に従ってsplit_data_balanceでデータを分割もしくはアンダーオーバーサンプリングする。
        X_test, Y_test がNoneでなければ検証データはこちらが優先されるが、Noneの場合
        split_data_balanceのindexを訓練と検証に使用して交差顕彰を行う
    fit_params: fit 時のパラメータ
    eval_params: 評価時のパラメータ
    """
    logger.info("START")

    # trial の設定. const とか int とかは自前で用意した解釈を簡単にするためのもの
    dict_input = {}
    for x in dict_param.keys():
        value = dict_param.get(x)
        if type(value) != list: logger.raise_error(f"value: {value} is not list.") # リスト形式でない場合はエラー
        if   value[0] == "const"   : dict_input[x] = value[1]
        elif value[0] == "int"     : dict_input[x] = trial.suggest_int(             x, value[1], value[2])
        elif value[0] == "float"   : dict_input[x] = trial.suggest_uniform(         x, value[1], value[2])
        elif value[0] == "log"     : dict_input[x] = trial.suggest_loguniform(      x, value[1], value[2])
        elif value[0] == "step"    : dict_input[x] = trial.suggest_discrete_uniform(x, value[1], value[2], value[3])
        elif value[0] == "category": dict_input[x] = trial.suggest_categorical(     x, value[1:])
    model_tuna = model.__class__(**dict_input)
    logger.info(model_tuna)

    train_indexs, test_indexs = [], []
    if split_params is not None and type(split_params) == dict:
        train_indexs, test_indexs = split_data_balance(Y, **split_params)
    
    def __Work(model_tuna, X, Y, fit_params, X_test, Y_test, eval_params):
        fit_params = conv_validdata_in_fitparmas(fit_params.copy(), X_test, Y_test)
        ## model fit
        model_tuna.fit(X, Y, **fit_params)
        ## model predict
        y_test_pred       = model_tuna.predict(X_test)
        y_test_pred_proba = None
        if is_callable(model_tuna, "predict_proba") == True:
            y_test_pred_proba = model_tuna.predict_proba(X_test)
        y_test_ans        = Y_test
        score = evalate(tuning_eval, y_test_ans, y_test_pred, y_pred_proba=y_test_pred_proba, **eval_params)
        return score

    score_list: np.ndarray = np.empty(0) # 結果の格納
    if len(train_indexs) >= 1:
        # split_data_balance が有効だった場合
        for train_index, test_index in zip(train_indexs, test_indexs):
            logger.info(f"cross validation {score_list.shape} ...")
            _validation_x, _validation_y = None, None
            if (X_test is not None and type(X_test) == np.ndarray):
                _validation_x = X_test
                _validation_y = Y_test
            else:
                _validation_x = X[test_index]
                _validation_y = Y[test_index]
            score = __Work(model_tuna, X[train_index], Y[train_index], fit_params, _validation_x, _validation_y, eval_params)
            score_list = np.append(score_list, score)
    else:
         # split_data_balance を使わなかった場合
        score = __Work(model_tuna, X, Y, fit_params, X_test, Y_test, eval_params)
        score_list = np.append(score_list, score)

    # 交差検証の平均値を返す
    return score_list.mean()


def search_hyperparams_by_optuna(
    optuna_study: optuna.study.Study, 
    model: object, X: np.ndarray, Y: np.ndarray, 
    n_trials: int=100, iters: int=None, X_test: np.ndarray=None, Y_test: np.ndarray=None, dict_param="auto", tuning_eval: str="rmse", 
    split_params: dict={"n_splits":1,"y_type":"cls","weight":"balance","is_bootstrap":False}, 
    fit_params: dict={}, eval_params: dict={}, n_jobs: int=1
) -> (pd.DataFrame, dict, ):
    """
    optuna でパラメータ探索
    Params::
        list_ouptut: 途中のデータも見たいので引数で空リスト
        n_trials: optuna で探索する回数
        iter: 木の深さ固定などで探索したい場合は指定する
        X: 訓練データ
        Y: 訓練正解ラベル
        X_test: テストデータ
        Y_test: テスト正解ラベル
        model: ハイパーパラメータサーチしたいモデル
        dict_param = {"learning_rate":["category",0.05,0.1,0.2,0.3,0.5], "n_estimators":["int",500,1500], "max_depth":["int",3,10], 
                    "gamma":["float",0.,0.3], "min_child_weight":["int",1,20], "subsample":["step", 0.5, 0.9, 0.1], 
                    "colsample_bytree":["step", 0.1, 0.9, 0.1], "random_state":["const",1], "n_jobs":["const", -1]}
        tuning_eval: kkpackage.util.features.evalate にあるtuning_eval
        split_params:
            この設定に従ってsplit_data_balanceでデータを分割もしくはアンダーオーバーサンプリングする。
            X_test, Y_test がNoneでなければ検証データはこちらが優先されるが、Noneの場合
            split_data_balanceのindexを訓練と検証に使用して交差顕彰を行う
        fit_params: fit 時のパラメータ
        eval_params: 評価時のパラメータ
    """
    if (type(dict_param) == str) and dict_param == "auto":
        ## 目的変数が2クラスか多クラスなのかを自動判断しておく
        bool_class_binary = True
        if is_classification_model(model):
            if np.unique(Y).shape[0] <= 2: bool_class_binary = True
            else:                          bool_class_binary = False
                
        if   str(type(model)).find("lightgbm.sklearn.LGBMClassifier") >= 0:
            dict_param = {
                "boosting_type"    :["category","gbdt","dart"], #"goss"
                "num_leaves"       :["int",10,1500],
                "max_depth"        :["const",-1],
                "learning_rate"    :["const",0.5], 
                "n_estimators"     :["const", 5000], 
                "subsample_for_bin":["const", 200000], 
                ## 必要に応じて変更する ‘binary’ or ‘multiclass'
                "objective"        :["const",("binary" if bool_class_binary==True else "multiclass")], 
                "class_weight"     :["const", "balanced"], 
                "min_child_weight" :["category"] + [0.01 * (2**i) for i in range(18)], 
                "min_child_samples":["int",1,1000], 
                "subsample"        :["step", 0.01, 0.99, 0.01], 
                "colsample_bytree" :["step", 0.01, 0.99, 0.01], 
                "reg_alpha"        :["const", 0],
                "reg_lambda"       :["category", 0] + [0.01 * (2**i) for i in range(18)],
                "random_state"     :["const",1], 
                "n_jobs"           :["const", n_jobs] 
            }
        elif str(type(model)).find("LGBMRegressor") >= 0:
            dict_param = {
                "boosting_type"    :["const","gbdt"], 
                "num_leaves"       :["int",10,1000],
                "max_depth"        :["const",-1], 
                "learning_rate"    :["const",0.3], 
                "n_estimators"     :["const", 3000], 
                "subsample_for_bin":["const", 200000], 
                "objective"        :["const","regression"], 
                "class_weight"     :["const", None], 
                "min_child_weight" :["float",0,100], 
                "min_child_samples":["int",1,100], 
                "subsample"        :["step", 0.01, 0.99, 0.01], 
                "colsample_bytree" :["step", 0.01, 0.99, 0.01], 
                "reg_lambda"       :["float",0,100], 
                "random_state"     :["const",1], 
                "n_jobs"           :["const", n_jobs] 
            }
        elif str(type(model)).find("xgboost.sklearn.XGBClassifier") >= 0:
            dict_param = {
                "n_estimators"     :["const", 3000], 
                "max_depth"        :["int", 3, 15], 
                "learning_rate"    :["const", 0.1], 
                "objective"        :["const",("binary:logistic" if bool_class_binary==True else "multi:softmax")], 
                "booster"          :["category", "gbtree","dart"], 
                "n_jobs"           :["const", n_jobs], 
                "gamma"            :["category", 0] + [0.01 * (2**i) for i in range(16)],
                "min_child_weight" :["category", 0] + [0.01 * (2**i) for i in range(16)],
                "max_delta_step"   :["const", 0], 
                "subsample"        :["step", 0.01, 0.99, 0.01], 
                "colsample_bytree" :["step", 0.01, 0.99, 0.01], 
                "colsample_bylevel":["const", 1],
                "colsample_bynode" :["const", 1],
                "reg_alpha"        :["category", 0] + [0.01 * (2**i) for i in range(16)],
                "reg_lambda"       :["category", 0] + [0.01 * (2**i) for i in range(16)],
                "random_state"     :["const",1] 
            }
        elif str(type(model)).find("HistGradientBoostingClassifier") >= 0:
            dict_param = {
                "loss"  :["category", "binary_crossentropy", "categorical_crossentropy"], 
                "learning_rate"    :["category",0.05,0.1,0.2,0.3,0.5], 
                "max_iter"         :["int",100,3000], 
                "max_leaf_nodes"   :["int",10,1000],
                "max_depth"        :["const", None], 
                "min_samples_leaf" :["int", 1, 100], 
                "max_bins"         :["const", 256], 
                "tol"              :["float", 1e-8, 1e-1], 
                "random_state"     :["const",1] 
            }
        elif str(type(model)).find("RandomForestClassifier") >= 0:
            dict_param = {
                "n_estimators"     :["int",100,3000], 
                "criterion"        :["category", "gini", "entropy"], 
                "max_depth"        :["int",2, 10], 
                "min_samples_split":["int", 1, 100], 
                "min_samples_leaf" :["int", 1, 100], 
                "max_features"     :["category", "auto", "log2", None, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5], 
                "max_leaf_nodes"   :["const", None], 
                "bootstrap"        :["const", True], 
                "random_state"     :["const",1], 
                "n_jobs"           :["const", n_jobs] 
            }
        elif str(type(model)).find("CatBoostClassifier") >= 0:
            dict_param = {
                "objective"           :["const","MultiClass"], 
                "custom_metric"       :["const","MultiClass"], 
                "eval_metric"         :["const","MultiClass"], 
                "n_estimators"        :["const", 5000], 
                "learning_rate"       :["const", 0.05], 
                "random_seed"         :["const", 1], 
                "reg_lambda"          :["float", 0, 100], 
                #"subsample"           :["step", 0.1, 1.0, 0.1], 
                "max_depth"           :["int", 2, 8], 
                #"min_child_samples"   :["int", 1, 100], 
                "nan_mode"            :["const", "Min"], 
                "task_type"           :["const", "GPU"], 
                "devices"             :["const", '0'] 
            }
        else:
            logger.raise_error(f"not supported model !! model:{str(type(model))}")
        # 深さを固定する
        if iters is not None:
            for x in ["n_estimators", "max_iter"]:
                if dict_param.get(x) is not None:
                    dict_param[x] = ["const", iters]
                    break
    # データの数を変えながら探索する
    df_optuna = pd.DataFrame()
    # 関数を埋め込む
    f = partial(
        optuna_base_function, X, Y, X_test, Y_test, model, 
        dict_param, tuning_eval, split_params, fit_params, eval_params
    )
    # ハイパーパラメータ探索
    optuna_study.optimize(f, n_trials=n_trials)

    # 結果を格納する
    for i_trial in optuna_study.trials:
        sewk = pd.Series(i_trial.params)
        sewk["value"]  = i_trial.value
        df_optuna = df_optuna.append(sewk, ignore_index=True)
    # パラメータを作成する
    dict_param_ret = {}
    for key, val in  dict_param.items():
        if val[0] == "const": dict_param_ret[key] = val[-1]
    for key, val in  optuna_study.best_params.items():
        dict_param_ret[key] = val

    logger.info("END")
    return df_optuna, dict_param_ret

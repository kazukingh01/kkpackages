import numpy as np
import pandas as pd
from typing import List
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error, mean_absolute_error, f1_score

# local package
from kkutils.util.numpy import multi_cross_entropy
from kkutils.util.com import is_callable, set_logger
logger = set_logger(__name__)


__all__ = [
    "is_classification_model",
    "predict_detail",
    "evalate",
    "confusion_matrix",
    "eval_classification_model",
    "eval_regressor_model"
]

def __accumulate_prediction(i, predict, X):
    prediction = predict(X, check_input=False)
    return [i, prediction]


def is_classification_model(model: object) -> bool:
    return is_callable(model, "predict_proba")


def predict_detail(model, X, do_estimators: bool=False, n_jobs: int=-1, **pred_params) -> pd.DataFrame:
    """
    model の予測結果を一通り計算して出力する
    Return::
        予測結果と予測確率をDataFrameにして返却する.
        回帰の場合は数値のみ
    Params::
        model: model
        X: 入力データ( np.ndarray or (np.ndarray, np.ndarray, ...) ) # マルチ入力でもそのまま渡ってくる
        do_estimators: 主に決定木の場合、各決定木の出力を取得するかどうか
        n_jobs: 並列数
    """
    if is_callable(model, "predict") == False:
        logger.raise_error(f"model :{model} doesn't have 'predict' method !")

    # 訓練データの精度を記録
    ## 結果の格納(訓練データと検証データの結果を格納する)
    ndf_pred = model.predict(X, **pred_params)
    if   len(ndf_pred.shape) == 1:
        df_score = pd.DataFrame(ndf_pred, columns=["predict"])
    elif len(ndf_pred.shape) == 2:
        df_score = pd.DataFrame(ndf_pred, columns=["predict_"+str(i) for i in range(ndf_pred.shape[1])])
    else:
        logger.raise_error(f"predict shape:{ndf_pred.shape} is over !!")
    
    ## 確率計算ができるかをチェックする
    ## classes_ には実際のラベルが入る(1,2のラベルを学習させると1,2で入る.0,1では入らない)
    if is_callable(model, "predict_proba") == True:
        logger.info("predict probability train dataset.")
        if is_callable(model, "classes_") == False: logger.raise_error(f"model: {model} doesn't have 'classes_' attr !")
        ndfwk = model.predict_proba(X, **pred_params)
        for i, label in enumerate(model.classes_.astype(int)):
            df_score["predict_proba_"+str(label)] = ndfwk[:, i]

    ## アンサンブル学習の場合、各弱学習機の結果を取得する
    if (do_estimators == True) and (is_callable(model, "estimators_") == True) and \
       (is_callable(model.estimators_, "predict", 0) == True):
        logger.info("predict estimators dataset.")
        out = Parallel(n_jobs=n_jobs, verbose=1)(delayed(__accumulate_prediction)(i_e, e.predict, X) \
                                                 for i_e, e in enumerate(model.estimators_))
        ## 結果を格納する。並列処理でずれた木のインデックスを整理する
        dfwk  = pd.DataFrame(out, columns=["i_tree","predict"])
        dfwk  = dfwk.sort_values(by="i_tree")
        ## ここのXはtupleかもしれないが、アンサンブルでは想定しないため.shapeを呼ぶ
        dfwk_ = pd.DataFrame(index=np.arange(X.shape[0]))
        for i_tree, ndfwk in dfwk[["i_tree","predict"]].values: dfwk_[i_tree] = ndfwk
        dfwk_["ensemble_predicts"] = dfwk_.apply(lambda _sewk: _sewk.values, axis=1)

        ## 結果を統合
        df_score["ensemble_predicts"] = dfwk_["ensemble_predicts"].values

    return df_score


def evalate(
        eval_method: str, y_ans: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray = np.array([]), 
        pos_label: int=(1, 1), eval_auc_list=None, **kwargs
    ):
    """
    評価用の関数(数値が低いほど良い評価値となるよう統一)
    Params::
        eval_method: roc_auc, roc_auc_multi, roc_auc_sum, accuracy, recall, precision, r2, rmse, mae
        y_ans: 正解ラベル
        y_pred: 予測値
        y_pred_proba: 各ラベルの予測確率
        pos_label: 正解ラベルとy_pred_probaの正解ラベルのあるindex
        eval_auc_list: eval_method = 'roc_auc_multi' の際、どのラベルを結合するかのpos_labelをリストにした版
                       list内のものを合計してaucを計算する List[(int, int, )]
    """
    logger.debug("START")
    score = None
    if   eval_method == "roc_auc":
        score = (1.0 - (roc_auc_score((y_ans == pos_label[0]), y_pred_proba[:, pos_label[1]])))
    elif eval_method == "roc_auc_multi":
        y_ans_wk        = np.zeros_like(y_ans).astype(bool)
        y_pred_proba_wk = np.zeros_like(y_ans)
        for label, i_label in eval_auc_list:
            y_ans_wk      = y_ans_wk | (y_ans == label)
            y_pred_proba += y_pred_proba[i_label]
        score = (1.0 - (roc_auc_score(y_ans_wk, y_pred_proba_wk)))
    elif eval_method == "roc_auc_sum":
        score = 0
        for i in range(y_pred_proba.shape[1]):
            score += (1.0 - (roc_auc_score((y_ans == i), y_pred_proba[:, i])))
    elif eval_method == "multi_logloss":
        score = multi_cross_entropy(y_pred_proba, y_ans, is_standardize=False).sum() # mean だと 0 の値も混ざるためダメ
    elif eval_method == "multi_logloss_with_std":
        score = multi_cross_entropy(y_pred_proba, y_ans, is_standardize=True).sum() # mean だと 0 の値も混ざるためダメ
    elif eval_method == "accuracy": score = (1.0 - ((y_ans == y_pred).sum() / y_ans.shape[0]))
    elif eval_method == "recall":
        score = 1.0 - (((y_ans == pos_label[0]) & (y_pred == pos_label[0])).sum() / (y_ans  == pos_label[0]).sum())
    elif eval_method == "precision":
        score = 1.0 - (((y_ans == pos_label[0]) & (y_pred == pos_label[0])).sum() / (y_pred == pos_label[0]).sum())
    elif eval_method == "f1":
        score = 1.0 - f1_score((y_ans == pos_label[0]), (y_pred == pos_label[0]))
    elif eval_method == "r2"   : score = -1.0 * (r2_score(y_ans, y_pred) - 1) ## 一番良くて1、悪いとマイナスになる
    elif eval_method == "rmse" : score = np.sqrt(mean_squared_error(y_ans, y_pred))
    elif eval_method == "mae"  : score = mean_absolute_error(y_ans, y_pred)
    logger.debug("END")
    return score


def confusion_matrix(df: pd.DataFrame, name_answer: str, name_predict: str, labels: List[int] = None) -> pd.DataFrame:
    """
    混同行列（Confusion Matrix）を作成する
    Params::
        df: input
        name_answer : 正解ラベルのカラム名
        name_predict: 予測ラベルのカラム名
        name_predict_proba: 予測確率のカラム名リスト
        labels: 正解ラベルの種類. None の場合関数内で求めるが、
                テストデータが少ない場合など正解ラベルが足りない場合があるので注意する
    """
    labels = [int(x) for x in np.sort(df[name_answer].unique())] if labels is None else labels
    df_conf = pd.DataFrame(index=labels)
    df_conf["count"] = df.groupby(name_answer).size()
    for x in labels: df_conf["predict_"+str(int(x))] = df[df[name_predict]==x].groupby(name_answer).size()
    df_conf = df_conf.fillna(0).astype(int)
    df_conf.index = ["answer_"+str(int(x)) for x in df_conf.index]
    return df_conf


def eval_classification_model(
        df: pd.DataFrame, name_answer: str, name_predict: str, name_predict_proba: List[str], 
        labels: List[int] = None, n_round: int=3, eval_auc_dict={}, **kwargs
    ) -> (pd.DataFrame, pd.Series):
    """
    分類モデルの一通りの評価値を計算して返却する
    Return::
        混合行列, 各評価値
    Params::
        df: input
        name_answer : 正解ラベルのカラム名
        name_predict: 予測ラベルのカラム名
        name_predict_proba: 予測確率のカラム名リスト
        labels: 正解ラベルの種類. None の場合関数内で求めるが、
                テストデータが少ない場合など正解ラベルが足りない場合があるので注意する
        n_round: 保存する数値のround
        eval_auc_dict: auc_multi の場合、どこをまとめるかのdictionary
            ex. {"name1":[(1,0,), (2,1,)], "name2":[(2,1,), (3,2,)]}
    """
    logger.info("START")
    se_eval = pd.Series(dtype=object)
    df   = df.copy()
    df[name_answer]  = df[name_answer].astype(int)
    df[name_predict] = df[name_predict].astype(int)
    y_ans  = df[name_answer].values
    y_pred = df[name_predict].values
    y_pred_proba = df[name_predict_proba].values
    labels = [int(x) for x in np.sort(df[name_answer].unique())] if labels is None else labels
    if y_pred_proba.shape[1] != len(labels):
        logger.raise_error(f"y_pred_proba: {y_pred_proba.shape}, labels: {labels}. Not match !")

    # 混同行列の作成
    df_conf = confusion_matrix(df, name_answer, name_predict, labels=labels)
    ## 各性能値
    se_eval["accuracy"] = evalate("accuracy", y_ans, y_pred)
    for x in labels: se_eval["recall_"   +str(x)] = evalate("recall",    y_ans, y_pred, pos_label=(x,0,))
    for x in labels: se_eval["precision_"+str(x)] = evalate("precision", y_ans, y_pred, pos_label=(x,0,))
    for x in labels: se_eval["f1_"       +str(x)] = evalate("f1",        y_ans, y_pred, pos_label=(x,0,))    

    ## aucの計算(多クラスの場合、全て2値化して計算する)
    for i_label, label in enumerate(labels):
        se_eval["roc_auc_"+str(label)] = evalate("roc_auc", y_ans, y_pred, y_pred_proba=y_pred_proba, pos_label=(label, i_label,))
    
    ## aucの計算. 多クラスの場合、任意に2値化した場合
    for name in eval_auc_dict.keys():
        eval_auc_list = eval_auc_dict.get(name)
        se_eval["roc_auc_"+name] = evalate("roc_auc_multi", y_ans, y_pred, y_pred_proba=y_pred_proba, eval_auc_list=eval_auc_list)

    ## 結果の返却(上位層で変数に格納させる)
    se_eval = se_eval.round(n_round)
    logger.info("END")
    return df_conf, se_eval


def eval_regressor_model(df, name_answer: str, name_predict: str, n_round: int=3, eval_cm_bin: dict=None, **kwargs):
    """
    分類モデルの一通りの評価値を計算して返却する
    Return::
        混合行列, 各評価値
    Params::
        df: input
        name_answer : 正解ラベルのカラム名
        name_predict: 予測ラベルのカラム名
        n_round: 保存する数値のraund
        eval_cm_bin: 回帰を分類のように扱って性能を得るときに使用
                     {1:[0,100]} のような形式で格納されており、list内の範囲(0-100)をkey(1)で置き換える
    """
    logger.info("START")
    se_eval = pd.Series()
    df   = df.copy()
    y_ans  = df[name_answer].values
    y_pred = df[name_predict].values

    ## 標準的な評価指標
    se_eval["R2"]   = evalate("r2", y_ans, y_pred)
    se_eval["RMSE"] = evalate("rmse", y_ans, y_pred)
    se_eval["MAE"]  = evalate("mae", y_ans, y_pred)

    # binningして混合行列を作成する(aucも)
    df_conf = pd.DataFrame()
    if type(eval_cm_bin) == dict:
        df["predict_bin"] = pd.np.nan # 初期化しておく
        df["answer_bin"]  = pd.np.nan # 初期化しておく
        ## {1:[0,100]} のような形式で格納されており、list内の範囲(0-100)をkey(1)で置き換える
        for x in eval_cm_bin.keys():
            listwk = eval_cm_bin.get(x)
            df.loc[((df[name_predict] >= listwk[0]) & (df[name_predict] < listwk[1])), "predict_bin"] = x
            df.loc[((df[name_answer]  >= listwk[0]) & (df[name_answer]  < listwk[1])), "answer_bin"]  = x

        # ラベルの一覧
        labels = [int(xx) for xx in np.sort(list(eval_cm_bin.keys()))]

        # 混同行列の作成
        df_conf = confusion_matrix(df, "answer_bin", "predict_bin", labels=labels)
        ## 比率も計算しておく
        y_ans  = df["answer_bin"].values
        y_pred = df["predict_bin"].values
        se_eval["accuracy"] = evalate("accuracy", y_ans, y_pred)
        for x in labels: se_eval["recall_"   +str(x)] = evalate("recall",    y_ans, y_pred, pos_label=(x,0,))
        for x in labels: se_eval["precision_"+str(x)] = evalate("precision", y_ans, y_pred, pos_label=(x,0,))
        for x in labels: se_eval["f1_"       +str(x)] = evalate("f1",        y_ans, y_pred, pos_label=(x,0,))    

    ## 結果の返却(上位層で変数に格納させる)
    se_eval = se_eval.round(n_round)
    logger.info("END")
    return df_conf, se_eval

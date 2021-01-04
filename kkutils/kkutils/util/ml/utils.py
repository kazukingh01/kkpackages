import pandas as pd
import numpy as np
from typing import List
from functools import partial
from joblib import Parallel, delayed
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

#  local package
from kkutils.util.dataframe import divide_index
from kkutils.util.com import set_logger
logger = set_logger(__name__)


__all__ = [
    "split_data_balance",
    "conv_validdata_in_fitparmas",
    "calc_randomtree_importance",
    "calc_mutual_information",
    "calc_parallel_mutual_information",
]


def split_data_balance(Y: np.ndarray, n_splits: int=1, y_type: str="cls", weight="balance", is_bootstrap :bool=False, random_seed: int=1) -> (List[np.ndarray], List[np.ndarray]):
    """
    交差検証用にデータを分割するためのインデックスを返却する
    Return:: (train_index, test_index, )
        train_index = [1分割目のindex, 2分割目のindex, ...]
        test_index  = [1分割目のindex, 2分割目のindex, ...]
    Params::
        Y: 正解ラベルのndarray
        n_splits: データを分割する数
        y_type: cls or reg. 分類か回帰
        weight: 分け方の定義
            balance, over, under, or {0:2, 1:"base", 2:0.5} or {0:100, 1:200, 2:100}
        is_bootstrap: bootstrap sampling するかどうか
        random_seed: random_seed
    """
    logger.info("START")
    np.random.seed(random_seed)

    def __sample(dictwk: dict, is_bootstrap: bool, n_data):
        dict_ret = {}
        if   type(n_data) == int:
            _n_data = n_data
            if is_bootstrap:
                for x in dictwk.keys():
                    dict_ret[x] = np.random.choice(dictwk[x], _n_data)
            else:
                for x in dictwk.keys():
                    if dictwk[x].shape[0] >= _n_data:
                        dict_ret[x] = dictwk[x][:_n_data]
                    else:
                        # np.array([1,2,3,4]) で n_data=10のとき 10 // 4 + 1 = 3 の回数増幅して np.array([1,2,3,4,1,2,3,4,1,2,3,4])を作り、10までのindexをとる
                        dict_ret[x] = np.hstack([dictwk[x] for i in range(_n_data // dictwk[x].shape[0] + 1)])[ : _n_data].copy()
        elif type(n_data) == dict:
            if is_bootstrap:
                for x in dictwk.keys():
                    _n_data = n_data[x]
                    dict_ret[x] = np.random.choice(dictwk[x], _n_data)
            else:
                for x in dictwk.keys():
                    _n_data = n_data[x]
                    if dictwk[x].shape[0] >= _n_data:
                        dict_ret[x] = dictwk[x][:_n_data]
                    else:
                        # np.array([1,2,3,4]) で n_data=10のとき 10 // 4 + 1 = 3 の回数増幅して np.array([1,2,3,4,1,2,3,4,1,2,3,4])を作り、10までのindexをとる
                        dict_ret[x] = np.hstack([dictwk[x] for i in range(_n_data // dictwk[x].shape[0] + 1)])[ : _n_data].copy()
        else:
            raise Exception("n_data is int or dict.")
        return dict_ret
    indexes, train_indexes, test_indexes = {}, [], []
    if y_type == "cls":
        # 分類の場合
        Y = Y.astype(int).copy()
        labels = np.sort(np.unique(Y))
        if type(weight) == dict:
            if [int(x) for x in sorted(list(weight.keys()))] != [int(x) for x in labels]:
                logger.raise_error(f"labels: {labels}, weight: {weight} is not enough keys.")
        indexes = {int(x):np.where(Y == x)[0] for x in labels} # label 毎のindexをdictで保持. 念の為intに変換. int64とかと区別するため
        indexes = {x: np.random.permutation(indexes.get(x)) for x in indexes.keys()} # 規則性を持ってインデックスが並んでいる可能性もあるため、ランダムに並べ替えておく
    elif y_type == "reg":
        # 回帰の場合
        ## sort してある範囲毎で分割する
        Y = Y.astype(float).copy()
        indexes = np.argsort(Y) # sort後のindexを返却する関数
        indexes = {i:indexes[(indexes.shape[0] // n_splits + 1) *  i :
                             (indexes.shape[0] // n_splits + 1) * (i+1)] for i in range(n_splits)} # 擬似的にクラスみたく分割する
    # まずテストデータのインデックスを決定する
    if n_splits >= 2:
        for i in range(n_splits):
            indexes_wk = {_x:indexes.get(_x)[(indexes.get(_x).shape[0] // n_splits) *  i: \
                                            (indexes.get(_x).shape[0] // n_splits) * (i+1) ] for _x in indexes.keys()}.copy()
            test_indexes.append(indexes_wk)
    else:
        ## n_splits = 1の場合、全てのインデックスがtest_indexesに振り分けられてしまうため、処理を分ける
        test_indexes.append({_x: np.array([]) for _x in indexes.keys() })
    # 次にテストデータのインデックス以外で訓練データのインデックスを決定する
    for i in range(n_splits):
        ## テストデータを含まないインデックス辞書
        indexes_except_test = {_x:indexes.get(_x)[~np.isin(indexes.get(_x), test_indexes[i][_x])] for _x in indexes.keys()}
        ## 上記をweightに従ってオーバー・アンダーサンプリングして訓練データのインデックスにする
        if   type(weight) == str and weight == "balance":
            train_indexes.append(indexes_except_test.copy())
        elif type(weight) == str and weight == "under":
            ## 一番数の少ないラベルに合わせる
            n_data = min([indexes_except_test[_x].shape[0] for _x in indexes_except_test.keys()])
            train_indexes.append(__sample(indexes_except_test, is_bootstrap, n_data))
        elif type(weight) == str and weight == "over":
            ## 一番数の多いラベルに合わせる
            n_data = max([indexes_except_test[_x].shape[0] for _x in indexes_except_test.keys()])
            train_indexes.append(__sample(indexes_except_test, is_bootstrap, n_data))
        elif type(weight) == dict:
            ## {0:2, 1:"base", 2:0.5} だとlabel=1 を基準にlabel=0は2倍, label=2は0.5倍とする
            ## {0:100, 1:200, 2:100} ののようにbaseがない場合は、その数までサンプリングする
            n_data_base = None
            for _x in weight.keys():
                if type(weight[_x]) == str and weight[_x] == "base":
                    n_data_base = indexes_except_test[_x].shape[0]
            ## base が設定されていたときだけ、weight を書き換える
            weight = weight.copy()
            if n_data_base is not None:
                for _x in weight.keys():
                    weight[_x] = n_data_base if (type(weight[_x]) == str and weight[_x] == "base") else int(weight[_x] * n_data_base)
            train_indexes.append(__sample(indexes_except_test, is_bootstrap, weight))
        else:
            logger.raise_error(f"weight: {weight} is error.")
    # dict を list に戻してrandom化
    train_indexes = [np.random.permutation(np.hstack(list(dictwk.values()))) for dictwk in train_indexes]
    test_indexes  = [np.random.permutation(np.hstack(list(dictwk.values()))) for dictwk in test_indexes ]
    logger.info("END")
    return train_indexes, test_indexes


def conv_validdata_in_fitparmas(fit_params, validation_x, validation_y):
    """
    fit_params 内の _validation_x, _validation_y を置き換える
    """
    logger.info("START")
    ret_fit_params = fit_params.copy()
    # "_validation_x", "_validation_y" を validation_x, validation_y に変換する
    for _x in fit_params.keys():
        ## validation data をtuple等で渡す場合がある. なので,文字列以外も考慮する
        if   type(fit_params.get(_x)) == str:
            if (fit_params.get(_x)   == "_validation_x"):
                ret_fit_params[_x] = validation_x
            elif (fit_params.get(_x) == "_validation_y"):
                ret_fit_params[_x] = validation_y
        elif type(fit_params.get(_x)) in [tuple, list]:
            ## list の list まで想定する
            ## tupleの場合は中身を上書きできないので、別で用意して詰め込む
            _tuple = []
            for _y in fit_params.get(_x):
                _tuple.append(None) #ひとまず空でつめる.
                if type(_y) == str:
                    if   _y == "_validation_x":
                        _tuple[-1] = validation_x
                    elif _y == "_validation_y":
                        _tuple[-1] = validation_y
                    else:
                        _tuple[-1] = _y
                elif type(_y) in [tuple, list]:
                    _tuplewk = []
                    for _z in _y:
                        _tuplewk.append(None)
                        if   type(_z) == str and _z == "_validation_x":
                            _tuplewk[-1] = validation_x
                        elif type(_z) == str and _z == "_validation_y":
                            _tuplewk[-1] = validation_y
                        else:
                            _tuplewk[-1] = _z
                    _tuple[-1] = tuple(_tuplewk) if type(_y) == tuple else _tuplewk
                else:
                    _tuple[-1] = _y
            ret_fit_params[_x] = _tuple
    logger.info(f'fit_params after this process:\n{ret_fit_params}')
    logger.info("END")
    return ret_fit_params


def calc_randomtree_importance(
    X: np.ndarray, Y: np.ndarray, colname_explain: np.ndarray, 
    is_cls_model: bool=True, n_estimators: int=100, cnt_thre: int=40, n_jobs: int=1
) -> pd.DataFrame:
    """
    ExtraTreesで重要度を計算する。完全ランダム決定木は変数選択も場合分けの場所もランダム
    Return::
        DataFrame. "feature_name","importance","std","count"
    Params::
        X: input
        Y: 正解ラベル
        colname_explain: 特徴量
        is_cls_model: モデルが分類かどうか
        n_estimators: 弱学習機の数
        cnt_thre: 各決定木に使われた特徴量の数の中央値がこの値を超えると計算を終える
        n_jobs: 並列数
    """
    logger.info("START")
    logger.info(f"input:{X.shape}, answer:{Y.shape}, colname_explain: {colname_explain}")

    # 特徴量スコア格納用DF
    df_features_cnt = pd.DataFrame(columns=colname_explain.copy())
    df_features_imp = pd.DataFrame(columns=colname_explain.copy())

    # 各特長量が一定回数、木の分岐で使われるまで繰り返す
    i = 0
    while(True):
        i += 1
        logger.info("create forest. loop:%s", i)
        # モデルの定義(木の数はとりあえず100固定)
        model = None
        dictwk = {"bootstrap":False, "n_estimators":n_estimators, "max_depth":10, "max_features":"auto", "verbose":3, "random_state":i, "n_jobs": n_jobs}
        if is_cls_model: model = ExtraTreesClassifier(**dictwk)
        else:            model = ExtraTreesRegressor(**dictwk)

        # モデルのFIT
        model.fit(X, Y)
        
        ## model内で特徴量を使用した回数をカウントする
        feature_count = np.hstack(list(map(lambda y: y.tree_.feature, model.estimators_)))
        feature_count = feature_count[feature_count >= 0] #-1以下は子がないとか特別の意味を持つ
        sewk = pd.DataFrame(colname_explain[feature_count], columns=[0]).groupby(0).size()
        df_features_cnt = df_features_cnt.append(sewk, ignore_index=True)

        ## modelの重要度を格納する
        sewk = pd.Series(model.feature_importances_, index=df_features_imp.columns.values.copy())
        df_features_imp = df_features_imp.append(sewk, ignore_index=True)

        logger.debug("\n%s", df_features_cnt)
        logger.debug("\n%s", df_features_imp)

        ## カウントが一定数達した場合に終了する
        ## ※例えば、ほとんどがnanのデータは分岐できる点が少ないためカウントが少なくなる
        cnt = df_features_cnt.sum(axis=0).median() #各特長量毎の合計の中央値
        logger.info("count median:%s", cnt)
        if cnt >= cnt_thre:
            break

    # 特徴量計算
    ## カウントがnanの箇所は、重要度もnanで置き換える(変換はndfを通して参照形式で行う)
    ## カウントが無ければnan. 重要度は、カウントがなければ0
    ndf_cnt = df_features_cnt.values.astype(np.float32)
    ndf_imp = df_features_imp.values.astype(np.float32)
    ndf_imp[np.isnan(ndf_cnt)] = np.nan #木の分岐にはあるが重要度として0がカウントする可能性も考慮して
    ## 重要度の計算
    df_features = df_features_imp.mean().reset_index().copy()
    df_features.columns  = ["feature_name","importance"]
    df_features["std"]   = df_features_imp.std().values #カラム順を変えていないので、joinしなくても良いはず
    df_features["count"] = df_features_cnt.sum().values
    df_features = df_features.sort_values("importance", ascending=False)

    logger.info("END")
    return df_features


def calc_mutual_information(ndfx: np.ndarray, ndfy: np.ndarray, *args, bins: int=100, base_max: int=1) -> (np.ndarray, np.ndarray):
    """
    相互情報量を計算する
    Params::
        ndfx, ndfy: 入力は0~base_maxに正規化されているとする
    """
    """ DEBUG
    import numpy as np
    from fast_histogram import histogram1d, histogram2d
    x = np.random.rand(1000, 20)
    y = np.random.rand(1000, 10)
    ndfx, ndfy, bins, base_max = x, y, 100, 1
    """
    from fast_histogram import histogram1d, histogram2d
    logger.info("START")
    list_ndf = []
    for x in ndfx.T:
        for y in ndfy.T:
            ndf = histogram2d(x, y, range=[[0, base_max], [0, base_max]], bins=bins)
            ndf = (ndf / ndf.sum()).astype(np.float16)
            list_ndf.append(ndf.reshape(1, *ndf.shape))
    ndf_xy  = np.concatenate(list_ndf, axis=0)
    ndf_x   = np.array([histogram1d(x, range=[0, base_max], bins=bins) for x in ndfx.T]) / ndfx.shape[0]
    ndf_y   = np.array([histogram1d(x, range=[0, base_max], bins=bins) for x in ndfy.T]) / ndfy.shape[0]
    ndf_x   = np.tile(np.tile(ndf_x.reshape(-1, bins, 1), bins), (1, ndfy.shape[1], 1)).reshape(-1, bins, bins)
    ndf_y   = np.tile(np.tile(ndf_y, bins).reshape(-1, bins, bins), (ndfx.shape[1], 1, 1))
    ndf_x_y = ndf_x * ndf_y
    elem: np.ma.core.MaskedArray = ndf_xy * np.ma.log(ndf_xy / ndf_x_y)
    val:  np.ma.core.MaskedArray = np.sum(elem * base_max/bins * base_max/bins, axis=(1,2))
    val = np.ma.filled(val, 0)
    index_x = np.tile(np.arange(ndfx.shape[1]).reshape(-1, 1), ndfy.shape[1]).reshape(-1)
    index_y = np.tile(np.arange(ndfy.shape[1]),                ndfx.shape[1]).reshape(-1)
    index_list = np.concatenate([[index_x], [index_y]], axis=0).T
    logger.info("END")
    return (index_list, val, *args)


def calc_parallel_mutual_information(df: pd.DataFrame, n_jobs: int=1, calc_size: int=100, bins: int=100, base_max: int=1):
    """
    相互情報量をDataFramebeベースで並列計算する. 
    DataFrameは正規化されている前提とする. 
    """
    n_div     = (df.shape[1] // calc_size) + 1
    list_cols = divide_index(df.columns.values, n_div=n_div, random=False)
    ndf_cols  = np.array(list_cols)
    list_cols = [(i, i+j, cols_x, cols_y, ) for i, cols_x in enumerate(list_cols) for j, cols_y in enumerate(list_cols[i:])]
    func = partial(calc_mutual_information, bins=bins, base_max=base_max)
    out_list  = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(delayed(func)(df[cols_x].values, df[cols_y].values, i, j) for i, j, cols_x, cols_y in list_cols)
    # index の変換
    listwk   = []
    for index_list, val, i, j in out_list:
        index_x = np.array(ndf_cols[i])
        index_y = np.array(ndf_cols[j])
        index_x = index_x[index_list[:, 0]]
        index_y = index_y[index_list[:, 1]]
        listwk.append((index_x, index_y, val))
    index_x = np.concatenate([x for x, _, _ in listwk], axis=0)
    index_y = np.concatenate([x for _, x, _ in listwk], axis=0)
    val     = np.concatenate([x for _, _, x in listwk], axis=0)
    # 値の格納
    df_mi = pd.DataFrame(np.nan, index=df.columns.values, columns=df.columns.values, dtype=np.float16)
    for x, y, z in zip(index_x, index_y, val):
        df_mi.loc[x, y] = z
    return df_mi
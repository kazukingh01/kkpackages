import pandas as pd
import numpy as np

# local package
from kkimagemods.util.logger import set_logger, set_loglevel
logger = set_logger(__name__)


def split_data_balance(Y: np.ndarray, n_splits: int=1, y_type: str="cls", weight="balance", is_bootstrap :bool=False, random_seed: int=1):
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
        labels    = np.sort(np.unique(Y))
        if type(weight) == dict:
            if [int(x) for x in sorted(list(weight.keys()))] != [int(x) for x in labels]:
                logger.raise_error(f"labels: {labels}, weight: {weight} is not enough keys.")
        indexes   = {int(x):np.where(Y == x)[0] for x in labels} # label 毎のindexをdictで保持. 念の為intに変換. int64とかと区別するため
        indexes   = {x: np.random.permutation(indexes.get(x)) for x in indexes.keys()} # 規則性を持ってインデックスが並んでいる可能性もあるため、ランダムに並べ替えておく
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



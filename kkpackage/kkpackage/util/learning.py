import re
from typing import List
import pandas as pd
import numpy as np
import torch
import lightgbm as lgb
from scipy import stats
from scipy.misc import derivative
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error, mean_absolute_error, f1_score
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

#  local package
from kkpackage.util.common import is_callable
from kkpackage.util.logger import set_logger
logger = set_logger(__name__)


def sort_each_columns(df: pd.DataFrame, **params) -> pd.DataFrame:
    dfret = pd.DataFrame(index=np.arange(df.shape[0]))
    for x in df.columns:
        logger.info(f"sort by:{x}")
        dfret[x] = df[x].copy().sort_values(**params).values
    return dfret


def search_features_by_variance(df: pd.DataFrame, cutoff: float=0.99, ignore_nan: bool=False, n_jobs: int = 1) -> np.ndarray:
    """
    各特徴量の値の重複度合いで特徴量をカットする
    Params::
        df: input DataFrame. 既に対象の特徴量だけに絞られた状態でinputする
        cutoff: 重複度合い. 0.99の場合、全体の数値の内99%が同じ値であればカットする
        ignore_nan: nanも重複度合いの一部とするかどうか
        n_jobs: 並列数. -1 は全CPU
    Return::
        カット後の特徴量リスト
    """
    logger.info("START")

    # まずは各列でsortする。nanは末尾に固まる
    # df.values を行うと、一番大きい型が ndf 全体に適用されるため非常に重い
    # そのため、各列を愚直にループする
    ## ループするのも遅いので並列化する
    _n_step = (n_jobs if n_jobs > 0 else 10)
    _step = df.columns.shape[0] // _n_step
    logger.info(f"parallel calculation, n_jobs:%{n_jobs}, n_step:{_n_step}, step:{_step}")
    out_list = Parallel(n_jobs=n_jobs, backend="loky", verbose=10) \
                    (delayed(sort_each_columns)(df[df.columns[(_i*_step):((_i+1)*_step)]]) for _i in range(_n_step+1))
    ## 並列計算後の結合処理
    columns_org = df.columns.copy()
    size_df = df.shape[0]
    df = None # メモリから消す
    df_con = pd.concat(out_list, axis=1, ignore_index=False, sort=False)
    df_con = df_con[columns_org] # 順番を整理する

    # 各列でcutoffのデータ数を求める
    sewk = pd.Series(int(df_con.shape[0]), index=df_con.columns.values)
    if ignore_nan == True:
        ## nanを考慮しない場合は、各列のnanの数だけ予め引いておく
        _sewk = df_con.isna().sum(axis=0)
        sewk  = sewk - _sewk
    sewk = (sewk * cutoff).astype(int)
    boolwk = np.array([False] * df_con.shape[1])

    # nanを考慮しない場合は早いアルゴリズムで回せるため、明確に処理を分割する
    if ignore_nan == True:
        logger.debug("ignore nan mode.")
        ## 各列ごとのループでは遅いので、nanの塊毎にループ
        ## 並列化処理をいろいろ検討したが、メモリ共有も複数プロセス化も遅かった
        ## 結局ループをやめるのが一番良い高速化。DFを一度に突き合わせて処理をさせる
        for n_data, sewkwk in sewk.reset_index().groupby(0)["index"]:
            logger.debug(f"n_data:{n_data}, target data:{sewkwk.shape[0]}, {sewkwk.values[0:1]}, ...")
            if n_data > 0:
                dfwk = df_con.loc[:, (df_con.columns.isin(sewkwk.values))].copy().dropna()
                dfwk_first = dfwk.iloc[:-1*n_data]
                dfwk_last  = dfwk.iloc[n_data]
                colname_list_wk = sewkwk[((dfwk_first.values == dfwk_last.values).sum(axis=0) > 0)]
                boolwk = boolwk | df_con.columns.isin(colname_list_wk)
            else:
                # 全てnanの列はdrop対象
                boolwk = boolwk | df_con.columns.isin(sewkwk)

    # nanを考慮しない場合
    else:
        logger.debug("Not ignore nan mode.")
        n_data = sewk.max()
        for i in np.arange(df_con.shape[0]):
            logger.debug(f"i_loop:{i}")
            if i+n_data >= size_df:
                break
            sewk_i_min = df_con.iloc[i]
            sewk_i_max = df_con.iloc[i+n_data]
            boolwk = boolwk | (sewk_i_min == sewk_i_max)

    logger.info("END")
    return columns_org[boolwk].values



def search_features_by_correlation(df: pd.DataFrame, cutoff: float=0.9, ignore_nan_mode: int=0, on_gpu_size: int=1, n_jobs: int=1) -> (pd.DataFrame, list, ):
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
        n_jobs: ignore_nan_mode=2 のときの並列数
    """
    logger.info("START")

    ## 相関係数の算出(標準化しなくても同じ結果)
    cut_list = np.array([])
    if ignore_nan_mode == 0:
        logger.info("ignore_nan_mode:0. if columns contain nan valus, return correlation value is nan.")        
        ndfwk = df.astype(np.float32).copy().values
        ndf_corr = np.corrcoef(ndfwk.T) # 逆行列を入れるのが正しい
        ## N×Nの行列になっている。下記領域■を0にしないと、cut_listの計算で狂う
        ## ■□□
        ## ■■□
        ## ■■■
        for _i in np.arange(ndf_corr.shape[1]): ndf_corr[_i:, _i] = np.nan
        ## cut対象を絞る
        cut_list = df.columns.values[( ((ndf_corr > cutoff) | (ndf_corr < -1*cutoff)).sum(axis=0) > 0 )]
    elif ignore_nan_mode == 1:
        logger.info("ignore_nan_mode:1. before calculation, fill nan to mean value.")        
        ## nanを平均値で埋めて、大雑把に計算するモード
        ndfwk = df.fillna(df.mean()).astype(np.float32).copy().values
        ndf_corr = np.corrcoef(ndfwk.T) # 逆行列を入れるのが正しい
        ## N×Nの行列になっている。下記領域を0にしないと、cut_listの計算で狂う
        for _i in np.arange(ndf_corr.shape[1]): ndf_corr[_i:, _i] = np.nan
        ## cut対象を絞る
        cut_list = df.columns.values[( ((ndf_corr > cutoff) | (ndf_corr < -1*cutoff)).sum(axis=0) > 0 )]            
    elif ignore_nan_mode == 2:
        logger.info("ignore_nan_mode:2. calculate pandas.corr() or corrwith().")        
        ## nanが入っていたら相関係数がnanになるので、pandasのlibで計算する
        cut_list = []
        df = df.copy() # もとの変数を変えないようにcopyする
        ## 各１列ずつでループすると非常に遅い。
        ## ただ、全列でcorr()しても処理が遅いため、ある程度塊で処理させる
        ## 相関係数を計算したかどうかを判定するmapを用意する
        df_map  = pd.DataFrame(True, index=df.columns.values, columns=df.columns.values)
        ## 対角以下は予めFalseにしておく(numpyにして処理した方が早い)
        ## ndf_map は df_map の参照であるため値の操作が可能。以降これを使う(早いため)
        ndf_map = df_map.values
        for i in np.arange(ndf_map.shape[0]): ndf_map[i:, i] = False

        ## 処理高速化のため、はじめにstep単位の塊でcorr()を計算し、後はcorrwith()で計算していく
        step = 100
        def __work(i, _df):
            logger.debug(f"job step1:{i}")
            return _df.corr()
        ## 処理対象を最初からstep抜き出す(clname_A と colname_Aは予めFalseにしているので重複しない)
        colname_list = []
        logger.debug("calculate pandas.corr()")
        for i in np.arange(0, (ndf_map.shape[0] // step)+1):
            colname_list_wk = df_map.columns[(i*step):((i+1)*step)].values
            if colname_list_wk.shape[0] == 0: break
            colname_list.append(colname_list_wk)
                # 計算予定の箇所を更新
            _tate = np.arange(df_map.shape[1])[df_map.index.  isin(colname_list_wk)]
            _yoko = np.arange(df_map.shape[1])[df_map.columns.isin(colname_list_wk)]
            ndf_map[np.ix_(_tate, _yoko)] = False
        # 並列処理
        _out = Parallel(n_jobs=n_jobs, backend="loky", verbose=10) \
                (delayed(__work)(_, df[ndfwk]) for _, ndfwk in enumerate(colname_list))
        for df_corr_wk in _out:
            ndf_corr_wk  = df_corr_wk.values
            ## 対角以下の成分をnanに変換
            for _i in np.arange(ndf_corr_wk.shape[0]): ndf_corr_wk[_i:, _i] = np.nan
            ## cut対象を絞る
            cut_list_wk = df_corr_wk.columns[(((ndf_corr_wk > cutoff) | (ndf_corr_wk < -1*cutoff)).sum(axis=0) > 0)].values.copy()
            cut_list.append(cut_list_wk)
            ## mapを更新(ndf_mapを更新して、参照しているdf_mapも更新される)
            ndf_map[:, df_map.columns.isin(cut_list_wk)   ] = False # cutした箇所
            ndf_map[   df_map.index.  isin(cut_list_wk), :] = False # cutした箇所
            logger.debug("remaining:%s, cut:%s", ndf_map.sum(), cut_list_wk)
            
        ## 相関係数をstep毎に計算する(while内部で並列化する)
        step = 1000 # 仮に1000とする
        def __work(i, _df, se):
            logger.debug(f"job step2:{i}")
            return _df.corrwith(se)
        logger.debug("calculate pandas.corrwith()")
        while(1):
            # 並列化のための並列数分のcolname_listを作成する
            colname_list = []
            for _ in range(n_jobs):
                ## 縦列からスキャン
                _bool = False
                for i in np.arange(ndf_map.shape[0]):
                    if ndf_map[i].sum() > 0:
                        ## Trueは未処理状態なので,その状態を見つけると抜ける
                        _bool = True
                        break
                if _bool == True:
                    ## 処理対象を最初からstep抜き出す(clname_A と colname_Aは予めFalseにしているので重複しない)
                    colname_list_wk = df_map.columns[df_map.iloc[i, :].values][:step].values
                    ndf_map[i, df_map.columns.isin(colname_list_wk)]   = False # 計算予定箇所
                    colname_list.append((i, colname_list_wk.copy()))
            # 一つも処理対象がなかったら,処理を抜ける
            if len(colname_list) == 0:
                break

            # 並列化
            _out = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)\
                    (delayed(__work)(_, df[tuplewk[1]], df[df_map.columns[tuplewk[0]]]) for _, tuplewk in enumerate(colname_list))
            ## cut対象を絞る
            for se_corr_wk in _out:
                cut_list_wk = se_corr_wk[((se_corr_wk > cutoff) | (se_corr_wk < -1*cutoff))].index.values.copy()
                cut_list.append(cut_list_wk)
            ## mapを更新(ndf_mapを更新して、参照しているdf_mapも更新される)
            ndf_map[:, df_map.columns.isin(cut_list_wk)   ] = False # cutした箇所
            ndf_map[   df_map.index.  isin(cut_list_wk), :] = False # cutした箇所
            logger.debug(f"target columns:{df_map.columns[colname_list[-1][0]]}, remaining:{ndf_map.sum()}, cut:{cut_list_wk}")

        cut_list_wk = []
        for x in cut_list: cut_list_wk = list(set(cut_list_wk + x.tolist()))
        cut_list = cut_list_wk
    elif ignore_nan_mode == 3:
        # GPUを使って計算する(GPUにのりきらないデータのために分割して計算する)
        ## メモリ不足解消のためランダムに分割してからastype(np.float32)する.内部でさらにfloat16で処理させる
        logger.info("ignore_nan_mode:3. Calculate with GPU.")
        _size = df.shape[0] // on_gpu_size
        ndf_index = np.random.permutation(np.arange(df.shape[0]))
        ndf_corr  = np.zeros(df.columns.values.shape[0]*df.columns.values.shape[0]).\
                    reshape(df.columns.values.shape[0], -1)
        for _i in range(on_gpu_size):
            logger.info(f"ignore_nan_mode:3. try: {_i}")
            ## 端数のデータは捨てても良い方針
            ndf = df.iloc[(ndf_index[(_size*_i):(_size*(_i+1))]), :].copy().values.astype(np.float32)
            ndf = corr_coef(ndf, _dtype=torch.float16).astype(np.float32)
            ndf_corr = ndf_corr + ndf
        ndf_corr = ndf_corr / float(on_gpu_size)
        ## N(特徴量)×Nの行列になっている。下記領域■を0にしないと、cut_listの計算で狂う
        ## ■□□
        ## ■■□
        ## ■■■
        for _i in np.arange(ndf_corr.shape[1]):
            ndf_corr[_i:, _i] = np.nan
        ## cut対象を絞る
        cut_list = df.columns.values[( ((ndf_corr > cutoff) | (ndf_corr < -1*cutoff)).sum(axis=0) > 0 )]

    # 相関係数を保存しておく
    df_corr = pd.DataFrame(ndf_corr, columns=df.columns.values, index=df.columns.values).astype(np.float16)
    return df_corr, cut_list


# 自作相関係数計算関数. GPUで高速化したい目的
def corr_coef(_ndf: np.ndarray, _dtype=torch.float16) -> np.ndarray:
    logger.info("START")
    # float16だと内積の計算で発散するので先に規格化する
    tensor_max  = torch.from_numpy(np.nanmax( _ndf, axis=0)).to(_dtype).to("cuda:0")
    tensor_min  = torch.from_numpy(np.nanmin( _ndf, axis=0)).to(_dtype).to("cuda:0")
    ndf = torch.from_numpy(_ndf).to(_dtype).to("cuda:0")
    ndf = (ndf - tensor_min) / tensor_max
    # nan 付きのvalueなのでnumpyでまずは処理する
    tensor_mean = torch.from_numpy(np.nanmean(ndf.cpu().numpy().astype(np.float32), axis=0)).to(_dtype).to("cuda:0")
    tensor_std  = torch.from_numpy(np.nanstd( ndf.cpu().numpy().astype(np.float32), axis=0)).to(_dtype).to("cuda:0")
    # 平均を引く
    tensor_Sxy = (ndf - tensor_mean).to(_dtype)
    # この時点でまずはnanのカウントをとる
    is_nan = torch.isnan(tensor_Sxy) # ここはboolean
    is_nan = torch.mm((~is_nan).t().half(), (~is_nan).half()).to(_dtype)
    # nanを0埋めしてから内積を計算する
    ## 下記の処理は一度cpuに落としてからやった方が早い
    _wk = torch.isnan(tensor_Sxy).to("cpu")
    tensor_Sxy = tensor_Sxy.to("cpu")
    tensor_Sxy[_wk] = 0
    ## GPUに戻す
    tensor_Sxy = tensor_Sxy.to("cuda:0")
    # 内積を計算する
    tensor_Sxy = (torch.mm(tensor_Sxy.t(), tensor_Sxy)).to(_dtype)
    tensor_Sxy = (tensor_Sxy / is_nan).to(_dtype)
    
    # 横軸分複製して、転置したものと掛けることによって分散×分散を計算する
    ## メモリが足らなくなるので一度CPUに戻して連結し、GPUに入れる
    tensor_SxSy = (torch.cat([tensor_std.reshape(1, -1) for i in range(ndf.shape[1])], dim=0)).to(_dtype)
    tensor_SxSy = (tensor_SxSy.t() * tensor_SxSy).to(_dtype)
    
    # numpyに戻すと少し計算がずれる(infがあればnanにしておく)
    ndf = (tensor_Sxy / tensor_SxSy).cpu().detach().numpy()
    ndf[ndf ==  np.inf] = np.nan
    ndf[ndf == -np.inf] = np.nan

    logger.info("END")
    return ndf


def search_features_by_kruskal(df: pd.DataFrame, colname_explain: List[str], colname_answer: str, cutoff: float=0.8):
    """クラスカル・ウォリス検定(Kruskal-Wallis test)"""
    logger.info("START")

    cut_list = []
    ## 説明変数でループ
    for i, x in enumerate(colname_explain):
        if i % 100 == 0: logger.debug(f"test loop... {i}")
        ndf = df.groupby(colname_answer)[x].apply(lambda y: y.values).values
        res = stats.kruskal(*ndf)
        if res.pvalue > cutoff: cut_list.append(x)

    # カットした特徴量の情報
    logger.info(f"cut features by correlation :{len(cut_list)}. {cut_list[:10]}")
    logger.info("END")
    return cut_list


def search_features_by_anova(df: pd.DataFrame, colname_explain: List[str], colname_answer: str, mode: str="mean", a=0.95):
    """
    分散分析(２次元以上は各ラベル中心値との距離で計算する)
    Paramas::
        modeは mean or median
    """
    logger.info("START")
    df = df[colname_explain + [colname_answer]].copy()

    ## 標準化
    standardscaler = StandardScaler()
    ndfwk = standardscaler.fit_transform(df[colname_explain].values)
    for i, x in enumerate(colname_explain): df[x] = ndfwk[:, i]

    ## ラベル毎の平均値を出すして列に追加する
    dfwk = pd.DataFrame()
    if   mode == "mean":   dfwk = df.groupby(colname_answer)[colname_explain].mean()
    elif mode == "median": dfwk = df.groupby(colname_answer)[colname_explain].median()
    else: logger.raise_esception("mode is 'mean' or 'median'.")
    for i, x in enumerate(colname_explain): df[x+"_mean"]      = df[colname_answer].apply(lambda y: dfwk.loc[y, x])
    for i, x in enumerate(colname_explain): df[x+"_mean_pow2"] = df[x+"_mean"].pow(2)
    for i, x in enumerate(colname_explain): df[x+"_mean_diff_pow2"] = (df[x] - df[x+"_mean"]).pow(2)
    
    # 偏差平方和 (sum of squares, SS)
    ## X は 各ラベル中心値と全体平均(標準化しているので0)との距離. Z は各ラベル要素と各ラベル中心値との距離
    X_SS = np.array([df[x+"_mean_pow2"     ].sum() for x in colname_explain]).sum()
    Z_SS = np.array([df[x+"_mean_diff_pow2"].sum() for x in colname_explain]).sum()
    X_MS = X_SS / (dfwk.shape[0] - 1)
    Z_MS = Z_SS / (df.shape[0] - dfwk.shape[0])
    F = X_MS / Z_MS # F値

    p = stats.f.ppf(a, dfwk.shape[0] - 1, df.shape[0] - dfwk.shape[0])
    logger.info("END")
    return F, p


def search_features_by_correlation_mic(df: pd.DataFrame, matrix_bins: int=100, cutoff: float=0.9, n_jobs: int=1):
    """非線形の相関分析mic(自作する)※区分けの実装が微妙なので没※"""
    logger.info("START")
    df = df.copy()

    # 特徴量を総当たりで探索する
    cut_list = []
    listwk = [(i,j) for i in range(df.columns.shape[0]) for j in range(i+1, df.columns.shape[0])]
    out = Parallel(n_jobs=n_jobs, verbose=1)(delayed(mic)(df[[df.columns[i], df.columns[j]]], matrix_bins) \
                                                    for i,j in listwk)
    out  = np.array(out)
    dfwk = pd.DataFrame(out, columns=["colname_x","colname_y","corr"])
    dfwk = dfwk[dfwk["corr"] > cutoff]
    cut_list = list(set(dfwk["colname_x"].tolist() + dfwk["colname_y"].tolist()))
                
    # カットした特徴量の情報
    logger.info(f"cut features by mic miccorrelation :{len(cut_list)}. {cut_list[:10]}")
    logger.info("END")
    return cut_list


# numpy形式(標準化はしない方が良い)での入力
# ※分割するbinの数が多いと１を超える。分割するgridもmicアルゴリズムがあるみたいなので、保留
def mic(df_, matrix_bins=100):
    logger.info("START")

    # ビンニングする前処理
    df = df_.apply(lambda x: (x - x.min()))
    df = df .apply(lambda x: x * matrix_bins * 0.999 / x.max())
    df = df.astype(int)
    df["work"] = 0
    df = df.pivot_table(values="work", index=df.columns[0], \
                        columns=df.columns[1], aggfunc="count").fillna(0)
    ndf = df.values
    n_samples = ndf.sum()
    Pxy = ndf / n_samples
    # x軸のカウント(縦の処理) ※tileは同じ配列を指定した方向に増やす
    Px  = np.tile(ndf.sum(axis=0), (df.shape[0],1)) / n_samples
    # x軸のカウント(縦の処理)
    Py  = np.tile(ndf.sum(axis=1), (df.shape[1],1)).T / n_samples
    # 情報量のマトリクス
    H = Pxy * np.log(Pxy/(Px*Py))

    logger.info("END")
    # mic相関係数の返却
    return np.array([df_.columns[0], df_.columns[1], np.nansum(H)])


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


def predict_detail(model, X, do_estimators: bool=False, n_jobs: int=-1, **kwargs) -> pd.DataFrame:
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
    df_score = pd.DataFrame(model.predict(X), columns=["predict"])
    
    ## 確率計算ができるかをチェックする
    ## classes_ には実際のラベルが入る(1,2のラベルを学習させると1,2で入る.0,1では入らない)
    if is_callable(model, "predict_proba") == True:
        logger.info("predict probability train dataset.")
        if is_callable(model, "classes_") == False: logger.raise_error(f"model: {model} doesn't have 'classes_' attr !")
        ndfwk = model.predict_proba(X)
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


# 並列処理で使うローカル関数の定義
def __accumulate_prediction(i, predict, X):
    prediction = predict(X, check_input=False)
    return [i, prediction]


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


def is_classification_model(model: object) -> bool:
    return is_callable(model, "predict_proba")


def conv_validdata_in_fitparmas(fit_params, validation_x, validation_y):
    """
    fit_params 内の _validation_x, _validation_y を置き換える
    """
    ret_fit_params = fit_params.copy()
    # "_validation_x", "_validation_y" を validation_x, validation_y に変換する
    for _x in fit_params.keys():
        ## validation data をtuple等で渡す場合がある. なので,文字列以外も考慮する
        if   type(fit_params.get(_x)) == str:
            if (fit_params.get(_x)   == "_validation_x"):
                ret_fit_params[_x] = validation_x
            elif (fit_params.get(_x) == "_validation_y"):
                ret_fit_params[_x] = validation_y
        elif type(fit_params.get(_x)) in [tuple, list] :
            ## tupleの場合は中身を上書きできないので、別で用意して詰め込む
            ## 受け取り手(model.fit())がlistの場合もtupleで渡してみる(それでエラーが発生するケースはないはず)
            _tuple = []
            for _i, _y in enumerate(fit_params.get(_x)):
                _tuple.append(None) #ひとまず空でつめる
                if   (type(_y) == str) and (_y == "_validation_x"):
                    _tuple[_i] = validation_x
                elif (type(_y) == str) and (_y == "_validation_y"):
                    _tuple[_i] = validation_y
            # _validation_x などは置き換えた値で埋めて、それ以外は元の値で埋める
            ret_fit_params[_x] = tuple(fit_params.get(_x)[_i] if _y is None else _y for _i, _y in enumerate(_tuple))
    return ret_fit_params


def calc_randomtree_importance(
        df: pd.DataFrame, colname_explain: np.ndarray, colname_answer: str, 
        is_cls_model: bool=True, n_estimators: int=100, cnt_thre: int=40, n_jobs: int=1
    ) -> pd.DataFrame:
    """
    ExtraTreesで重要度を計算する。完全ランダム決定木は変数選択も場合分けの場所もランダム
    Return::
        DataFrame. "feature_name","importance","std","count"
    Params::
        df: input
        colname_explain: 特徴量
        colname_answer: 正解ラベル
        is_cls_model: モデルが分類かどうか
        n_estimators: 弱学習機の数
        cnt_thre: 各決定木に使われた特徴量の数の中央値がこの値を超えると計算を終える
        n_jobs: 並列数
    """
    logger.info("START")
    logger.info(f"input:{df.shape}, colname_explain: {colname_explain}, colname_answer: {colname_answer}")
    # 正解ラベルを先に退避
    sewk = df[colname_answer].copy()

    # nanを最初に各特徴量毎の最小値と最大値のランダムで埋める
    df = df[colname_explain].astype(np.float32)
    df = df.replace(np.inf, np.nan).replace(-1*np.inf, np.nan) #infは先に変換
    ndfwk = np.random.permutation(np.arange(df.shape[0]))
    ndfwk1, ndfwk2 = ndfwk[:ndfwk.shape[0]//2], ndfwk[ndfwk.shape[0]//2:]
    dfwk1 = df.apply(lambda x: x.fillna(x.min() - 2)).iloc[ndfwk1] #各特長量毎の最小値-2で埋める
    dfwk2 = df.apply(lambda x: x.fillna(x.max() + 2)).iloc[ndfwk2] #各特長量毎の最大値+2で埋める
    df = pd.concat([dfwk1, dfwk2], axis=0, sort=False, ignore_index=False) #maxとminをブレンドする
    dfwk1, dfwk2 = None, None # メモリの節約
    df = df.fillna(0) #それでも埋まらないnan(つまり、列の全てがnan値)は0埋めする
    
    # 学習用のndf
    X, Y = df.values, None # ここで新規にfitさせるため、前処理の反映などは必要ない
    if is_cls_model: Y = sewk.loc[df.index].astype(np.int32).values
    else:            Y = sewk.loc[df.index].astype(np.float32).values
        
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


# あるグループ間での特徴量を作成する
## df_train に上書きする
def create_features_by_basic_method(
        df: pd.DataFrame, colname_group: str, colname_group_regex, replace_inf: np.float=np.nan,
        calc_list=["sum","mean","std","max","min","rank","diff","ratio"]
    ) -> pd.DataFrame:
    """
    あるグループ間での特徴量を作成する
    Params::
        df: input
        colname_group: 新規に特徴量を作成する際の先頭カラム名. colname_group_sum となる
        colname_group_regex: カラム名を選択する正規表現かリスト
        calc_list: 新規の特徴量作成の種類のリスト["sum","mean","std","max","min","rank","diff","ratio"]
    """
    # 特徴量作成するグループ項目の定義
    colname_org = np.empty(0)
    if type(colname_group_regex) == str:
        colname_org = df.columns[df.columns.str.contains(colname_group_regex, regex=True)].values
    elif type(colname_group_regex) == list:
        for x in colname_group_regex:
            _colname_wk = df.columns[df.columns.str.contains(x, regex=True)].values
            colname_org = np.append(colname_org, _colname_wk)

    # あるグループだけの特徴量でDFを定義する
    df = df[colname_org].copy()

    # グループ内を要約する
    if "sum"  in calc_list: df[colname_group + "_sum"]  = df[colname_org].sum(axis=1)
    if "mean" in calc_list: df[colname_group + "_mean"] = df[colname_org].mean(axis=1)
    if "std"  in calc_list: df[colname_group + "_std"]  = df[colname_org].std(axis=1)
    if "max"  in calc_list: df[colname_group + "_max"]  = df[colname_org].max(axis=1)
    if "min"  in calc_list: df[colname_group + "_min"]  = df[colname_org].min(axis=1)

    # ランキングする
    dfwk = df[colname_org].rank(axis=1) #, method='first'
    for x in dfwk.columns:
        if "rank" in calc_list: df[colname_group+"_"+x+"_rank"] = dfwk[x]

    # 各項目間の差・比率を計算する
    for i, x in enumerate(colname_org):
        for y in colname_org[i:]:
            if x == y: continue
            if "diff"  in calc_list: df[colname_group+"_"+x+"_"+y+"_diff" ] = df[x] - df[y]
            if "ratio" in calc_list: df[colname_group+"_"+x+"_"+y+"_ratio"] = df[x] / df[y]

    # inf の項目を変換する
    if type(replace_inf) is not None:
        ndfwk = np.isinf(df.values).sum(axis=0)
        ndfwk = (ndfwk > 0) # inf の数が0以上
        colname_wk = df.loc[:, ndfwk].columns.values
        for _x in colname_wk:
            df.loc[np.isinf(df[_x].values), _x] = replace_inf    
            
    # 追加した特徴量を反映
    colname_add = df.columns[~df.columns.isin(colname_org)].values

    # 追加した特徴量の型を適切に修正する
    df_ret = df[colname_add].copy()
    colname_bool = np.array([len(re.findall("(_sum|_max|_min|_diff)$", x))>0 for x in colname_add])
    if (df[colname_org].dtypes == np.float64).sum() > 0:
        df_ret = df[colname_add].astype(np.float64)
    elif (df[colname_org].dtypes == np.float32).sum() > 0:
        df_ret = df[colname_add].astype(np.float32)
    elif (df[colname_org].dtypes == np.float16).sum() > 0:
        df_ret = df[colname_add].astype(np.float16)
    elif (df[colname_org].dtypes == np.int64).sum() > 0:
        if colname_bool.sum() > 0:
            df_ret = df[colname_add[colname_bool]].astype(np.int64)
            if (~colname_bool).sum() > 0: df_ret[colname_add[~colname_bool]] = df[colname_add[~colname_bool]].astype(np.float32)
        else:
            df_ret = df[colname_add[~colname_bool]].astype(np.int64)
    elif (df[colname_org].dtypes == np.int32).sum() > 0:
        if colname_bool.sum() > 0:
            df_ret = df[colname_add[colname_bool]].astype(np.int32)
            if (~colname_bool).sum() > 0: df_ret[colname_add[~colname_bool]] = df[colname_add[~colname_bool]].astype(np.float32)
        else:
            df_ret = df[colname_add[~colname_bool]].astype(np.int32)
    elif (df[colname_org].dtypes == np.int16).sum() > 0:
        if colname_bool.sum() > 0:
            df_ret = df[colname_add[colname_bool]].astype(np.int16)
            if (~colname_bool).sum() > 0: df_ret[colname_add[~colname_bool]] = df[colname_add[~colname_bool]].astype(np.float32)
        else:
            df_ret = df[colname_add[~colname_bool]].astype(np.int16)
    elif (df[colname_org].dtypes == np.int8).sum() > 0:
        if colname_bool.sum() > 0:
            df_ret = df[colname_add[colname_bool]].astype(np.int8)
            if (~colname_bool).sum() > 0: df_ret[colname_add[~colname_bool]] = df[colname_add[~colname_bool]].astype(np.float16)
        else:
            df_ret = df[colname_add[~colname_bool]].astype(np.int8)
    ## rank は必ずfloat16とする
    colname_bool = np.array([len(re.findall("_rank$", x))>0 for x in colname_add])
    if colname_bool.sum() > 0:
        df_ret[colname_add[colname_bool]] = df[colname_add[colname_bool]].astype(np.float16)
    
    return df_ret
    

def softmax(x):
    f = np.exp(x)/np.sum(np.exp(x), axis = 1, keepdims = True)
    return f

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def rmse(x: np.ndarray, t: np.ndarray):
    return (x - t) ** 2 / 2.

def binary_cross_entropy(x: np.ndarray, t: np.ndarray):
    t = t.astype(np.int32)
    x = sigmoid(x)
    return -1 * (t * np.log(x) + (1 - t) * np.log(1 - x))

def multi_cross_entropy(x: np.ndarray, t: np.ndarray):
    t = t.astype(np.int32)
    if len(x.shape) > 1:
        x = softmax(x) # softmax で確率化
        t = np.identity(x.shape[1])[t]
        return -1 * t * np.log(x)
    else:
        x = sigmoid(x)
        x[t == 0] = 1 - x[t == 0] # 0ラベル箇所は確率を反転する
        return -1 * np.log(x)

def focal_loss(x: np.ndarray, t: np.ndarray, gamma: float=1) -> np.ndarray:
    """
    Params::
        x:
            予測値. 規格化されていない値であること. 1次元
            >>> x
            array([0.65634349, 0.8510698 , 0.61597224])
            multi classのとき. 例えば 3 class の場合は下記のような値になっていること.
            >>> x
            array([
                [0.65634349, 0.8510698 , 0.61597224],
                [0.58012161, 0.79659195, 0.39168051]
            ])
        t:
            正解値.
            multi classのとき. 例えば 3 class の場合は [0,0,0,1,1,1,2,1,2,0,0,...](0〜2まで)
        multi class に対してどうするか。categorycal cross entropy と同じ考え方で、正解ラベル以外の列はlossが0になる
        ※このアルゴリズムテクニックだけ残しておく。今は使っていない
        ※x[np.arange(t.shape[0]).reshape(-1, 1), t.reshape(-1, 1)] = 1 - x[np.arange(t.shape[0]).reshape(-1, 1), t.reshape(-1, 1)]
    """
    t = t.astype(np.int32)
    if len(x.shape) > 1:
        x = softmax(x) # softmax で確率化
        t = np.identity(x.shape[1])[t]
        return -1 * t * (1 - x)**gamma * np.log(x)        
    else:
        x = sigmoid(x)
        x[t == 0] = 1 - x[t == 0] # 0ラベル箇所は確率を反転する
        return -1 * (1 - x)**gamma * np.log(x)

def focal_loss_grad(x: np.ndarray, t: np.ndarray, gamma: float=1) -> (np.ndarray, np.ndarray, ):
    """
    内部に softmax を含む関数については derivative では計算が安定しない.
    かなり面倒だが、真面目に微分してみる。参考 https://hackmd.io/OddWU6zlR2GkrZNsl4IPnA
    """
    t = t.astype(np.int32)
    if len(x.shape) > 1:
        x = softmax(x) # softmax で確率化
        # 正解列を抜き出し
        xK = x[np.arange(t.shape[0]).reshape(-1, 1), t.reshape(-1, 1)]
        xK = np.tile(xK, (1, x.shape[1]))
        # x1 は 不正解列に -1 をかけて、さらに正解列はそこから1を足す操作
        x1 = x.copy()
        x1 = -1 * x1
        x1[np.arange(t.shape[0]).reshape(-1, 1), t.reshape(-1, 1)] = x1[np.arange(t.shape[0]).reshape(-1, 1), t.reshape(-1, 1)] + 1
        dfdy = gamma * (1 - xK) ** (gamma-1) * np.log(xK) - ((1 - xK) ** gamma / xK)
        dydx = xK * x1
        grad = dfdy * dydx

        dfdydx = dydx * (2 * gamma * (1 - xK) ** (gamma - 1) / xK - gamma * (gamma - 1) * np.log(xK) * (1 - xK) ** (gamma - 2) + (1 - xK) ** gamma * (xK ** -2))
        dydxdx = dydx * (1 - 2 * x)
        hess = dfdy * dydxdx + dydx * dfdydx
    else:
        grad = derivative(lambda _x: focal_loss(_x, t, gamma=gamma), x, n=1, dx=1e-6)
        hess = derivative(lambda _x: focal_loss(_x, t, gamma=gamma), x, n=2, dx=1e-6)

    return grad, hess


def calc_grad_hess(x: np.ndarray, t: np.ndarray, loss_func, dx=1e-6, **kwargs) -> (np.ndarray, np.ndarray, ):
    logger.debug(f'dx: {dx}, loss: {loss_func(x, t, **kwargs)}')
    grad = derivative(lambda _x: loss_func(_x, t, **kwargs), x, n=1, dx=dx)
    hess = derivative(lambda _x: loss_func(_x, t, **kwargs), x, n=2, dx=dx)
    return grad, hess


def lgb_custom_objective(y_pred: np.ndarray, data: lgb.Dataset, func_loss, is_lgbdataset: bool=True):
    """
    lightGBMのcustomized objectiveの共通関数
    Params::
        y_pred:
            予測値. multi classの場合は、n_sample * n_class の長さになったいる
            値は、array([0データ目0ラベルの予測値, ..., Nデータ目0ラベルの予測値, 0データ目1ラベルの予測値, ..., ])
        data:
            train_set に set した値
        func_loss:
            y_pred, y_true を入力に持ち、y_pred と同じ shape を持つ return をする
        is_lgbdataset:
            lgb.dataset でなかった場合は入力が逆転するので気をつける
    """
    if is_lgbdataset == False:
        y_true = y_pred.copy()
        y_pred = data
    else:
        y_true = data.label
    if y_pred.shape[0] != y_true.shape[0]:
        # multi class の場合
        n_class = int(y_pred.shape[0] / y_true.shape[0])
        y_pred = y_pred.reshape(n_class, -1).T
    logger.debug(f"y_pred: {y_pred}")
    logger.debug(f"y_true: {y_true}")
    grad, hess = func_loss(y_pred, y_true)
    logger.debug(f"grad: {grad}")
    logger.debug(f"hess: {hess}")
    return grad.T.reshape(-1), hess.T.reshape(-1)


def lgb_custom_eval(y_pred: np.ndarray, data: lgb.Dataset, func_loss, func_name: str, is_higher_better: bool, is_lgbdataset: bool=True):
    """
    lightGBMのcustomized objectiveの共通関数
    Params::
        y_pred:
            予測値. multi classの場合は、n_sample * n_class の長さになったいる
            値は、array([0データ目0ラベルの予測値, ..., Nデータ目0ラベルの予測値, 0データ目1ラベルの予測値, ..., ])
        data:
            train_set に set した値
        func_loss:
            y_pred, y_true を入力に持ち、grad, hess を return する関数
    """
    if is_lgbdataset == False:
        y_true = y_pred.copy()
        y_pred = data
    else:
        y_true  = data.label
    n_class = 1
    if y_pred.shape[0] != y_true.shape[0]:
        # multi class の場合
        n_class = int(y_pred.shape[0] / y_true.shape[0])
        y_pred = y_pred.reshape(n_class, -1).T
    value = func_loss(y_pred, y_true)
    return func_name, np.sum(value), is_higher_better

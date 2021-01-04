import re
import numpy as np
import pandas as pd
from typing import List
from joblib import Parallel, delayed
from scipy import stats

# local package
from kkutils.util.dataframe import sort_each_columns
from kkutils.util.torch import corr_coef, corr_coef_2ndarray
from kkutils.util.com import set_logger
logger = set_logger(__name__)


__all__ = [
    "search_features_by_variance",
    "search_features_by_correlation",
    "search_features_by_kruskal",
    "search_features_by_anova",
    "search_features_by_correlation_mic",
    "create_features_by_basic_method",
]


def sort_each_columns_parallel(df: pd.DataFrame, n_div: int=1, n_jobs: int=1) -> pd.DataFrame:
    logger.info("START")
    logger.info(f"df shape: {df.shape}, n_div:{n_div}, n_jobs:{n_jobs}")
    _step = df.columns.shape[0] // n_div
    out_list = Parallel(n_jobs=n_jobs, backend="loky", verbose=10) \
                    (delayed(sort_each_columns)(df[df.columns[(_i*_step):((_i+1)*_step)]]) for _i in range(n_div+1))
    ## 並列計算後の結合処理
    columns_org = df.columns.copy()
    del df # メモリから消す
    df_con = pd.concat(out_list, axis=1, ignore_index=False, sort=False)
    df_con = df_con[columns_org] # 順番を整理する
    logger.info("END")
    return df_con


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
    columns_org = df.columns.copy()
    # まずは各列でsortする。nanは末尾に固まる
    # df.values を行うと、一番大きい型が ndf 全体に適用されるため非常に重い
    # そのため、各列を愚直にループする
    ## ループするのも遅いので並列化する
    size_df = df.shape[0]
    n_div   = (n_jobs if n_jobs > 0 else 10)
    df_con  = sort_each_columns_parallel(df, n_div=n_div, n_jobs=n_jobs)
    del df # メモリから消す
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
            boolwk = boolwk | (sewk_i_min == sewk_i_max).values | (sewk_i_min.isna() & sewk_i_max.isna()).values # nan == nan が Falseになる事を考慮する

    logger.info("END")
    return columns_org[boolwk].values


def search_features_by_correlation(df: pd.DataFrame, cutoff: float=0.9, ignore_nan_mode: int=0, n_div_col: int=1, on_gpu_size: int=1, min_n_not_nan: int=10, n_jobs: int=1) -> (pd.DataFrame, list, ):
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
                n_div_col: columns が大きすぎるの行列計算でメモリに乗らないため、columns を分割する. 10000 を超えると難しい
                on_gpu_size: ignore_nan_mode=3のときに使う. 行列が全てGPUに乗り切らないときに、何分割するかの数字
                min_n_nan: ignore_nan_mode=3のときに使う. 
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
        _cols = df.shape[1] // n_div_col
        ndf_index = np.random.permutation(np.arange(df.shape[0]))
        ndf_corr  = np.zeros((df.shape[1], df.shape[1]), dtype=np.float16).astype(np.float16)
        if n_div_col > 1:
            for _i in range(on_gpu_size):
                pattern = [(j,i,) for i in range(n_div_col) for j in range(i+1)]
                for _j, _k in pattern:
                    logger.info(f"ignore_nan_mode:3. try: {_i}, {_j}, {_k}")
                    ## 端数のデータは捨てても良い方針
                    ndf1 = df.iloc[(ndf_index[(_size*_i):(_size*(_i+1))]), _j*_cols:( (_j+1)*_cols if _j != (n_div_col-1) else df.shape[1] )].copy().values.astype(np.float32)
                    ndf2 = df.iloc[(ndf_index[(_size*_i):(_size*(_i+1))]), _k*_cols:( (_k+1)*_cols if _k != (n_div_col-1) else df.shape[1] )].copy().values.astype(np.float32)
                    ndf  = corr_coef_2ndarray(ndf2, ndf1, _dtype="float16", min_n_not_nan=min_n_not_nan).astype(np.float16)
                    ndf_corr[_j*_cols:( (_j+1)*_cols if _j != (n_div_col-1) else df.shape[1] ), _k*_cols:( (_k+1)*_cols if _k != (n_div_col-1) else df.shape[1] )] += ndf
                    del ndf1, ndf2, ndf
        else:
            for _i in range(on_gpu_size):
                logger.info(f"ignore_nan_mode:3. try: {_i}")
                ## 端数のデータは捨てても良い方針
                ndf = df.iloc[(ndf_index[(_size*_i):(_size*(_i+1))]), :].copy().values.astype(np.float32)
                ndf = corr_coef(ndf, _dtype="float16", min_n_not_nan=min_n_not_nan).astype(np.float16)
                ndf_corr = ndf_corr + ndf
        ndf_corr = ndf_corr / np.float16(on_gpu_size)
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
    logger.info("END")
    return df_corr, cut_list



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
    from sklearn.preprocessing import StandardScaler
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


def _mic(df_: pd.DataFrame, matrix_bins: int=100) -> np.ndarray:
    """
    numpy形式(標準化はしない方が良い)での入力
    ※分割するbinの数が多いと１を超える。分割するgridもmicアルゴリズムがあるみたいなので、保留
    """
    # ビンニングする前処理
    df = df_.apply(lambda x: (x - x.min()))
    df = df .apply(lambda x: x * matrix_bins * 0.999 / x.max())
    df = df.astype(int)
    df["work"] = 0
    df = df.pivot_table(values="work", index=df.columns[0], columns=df.columns[1], aggfunc="count").fillna(0)
    ndf = df.values
    n_samples = ndf.sum()
    Pxy = ndf / n_samples
    # x軸のカウント(縦の処理) ※tileは同じ配列を指定した方向に増やす
    Px  = np.tile(ndf.sum(axis=0), (df.shape[0],1)) / n_samples
    # x軸のカウント(縦の処理)
    Py  = np.tile(ndf.sum(axis=1), (df.shape[1],1)).T / n_samples
    # 情報量のマトリクス
    H = Pxy * np.log(Pxy/(Px*Py))
    # mic相関係数の返却
    return np.array([df_.columns[0], df_.columns[1], np.nansum(H)])


def search_features_by_correlation_mic(df: pd.DataFrame, matrix_bins: int=100, cutoff: float=0.9, n_jobs: int=1):
    """非線形の相関分析mic(自作する)※区分けの実装が微妙なので没※"""
    logger.info("START")
    df = df.copy()

    # 特徴量を総当たりで探索する
    cut_list = []
    listwk = [(i,j) for i in range(df.columns.shape[0]) for j in range(i+1, df.columns.shape[0])]
    out = Parallel(n_jobs=n_jobs, verbose=1)(delayed(_mic)(df[[df.columns[i], df.columns[j]]], matrix_bins) \
                                                    for i,j in listwk)
    out  = np.array(out)
    dfwk = pd.DataFrame(out, columns=["colname_x","colname_y","corr"])
    dfwk = dfwk[dfwk["corr"] > cutoff]
    cut_list = list(set(dfwk["colname_x"].tolist() + dfwk["colname_y"].tolist()))
                
    # カットした特徴量の情報
    logger.info(f"cut features by mic miccorrelation :{len(cut_list)}. {cut_list[:10]}")
    logger.info("END")
    return cut_list


def create_features_by_basic_method(
    df: pd.DataFrame, colname_group: str, colname_group_regex, replace_inf: np.float=np.nan,
    calc_list=["sum","mean","std","max","min","rank","diff","ratio"],
) -> pd.DataFrame:
    """
    あるグループ間での特徴量を作成する
    Params::
        df: input
        colname_group: 新規に特徴量を作成する際の先頭カラム名. colname_group_sum となる
        colname_group_regex: カラム名を選択する正規表現かリスト
        calc_list: 新規の特徴量作成の種類のリスト["sum","mean","std","max","min","rank","diff","ratio"]
    """
    logger.info("START")
    # 特徴量作成するグループ項目の定義
    colname_org = np.empty(0)
    if colname_group_regex is None:
        logger.warning(f"all columns is target of new features. columns: {df.columns}")
        colname_org = df.columns.values
    else:
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
    if "rank" in calc_list:
        dfwk = df[colname_org].rank(axis=1) #, method='first'
        for x in dfwk.columns:
            df[colname_group+"_"+x+"_rank"] = dfwk[x].copy()
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
    logger.info("END")
    return df_ret

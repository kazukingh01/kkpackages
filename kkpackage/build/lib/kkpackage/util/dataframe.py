import pandas as pd
import numpy as np
import logging
from typing import List

# local
from kkpackage.util.common import check_type, correct_dirpath

def values_include(target, check_list) -> np.ndarray:
    """
    target に check_list の value が全てあるか調べる
    Return::
        check_list のうち、target に含まれている value のリストを返す
    Params::
        target: df の columns とか
        check_list: 全て含まれていてほしい値のリスト. 1次元
    """
    target     = conv_pd_index(target).copy()
    check_list = conv_pd_index(check_list).copy()
    return check_list[np.isin(check_list, target)]

def values_not_include(target, check_list) -> np.ndarray:
    """
    target に check_list の value が全てあるか調べる
    Return::
        check_list のうち、target に含まれていない value のリストを返す
    Params::
        target: df の columns とか
        check_list: 全て含まれていてほしい値のリスト. 1次元
    """
    target     = conv_pd_index(target).copy()
    check_list = conv_pd_index(check_list).copy()
    return check_list[~check_list.isin(target)]

def conv_ndarray(target) -> np.ndarray:
    check_type(target, [list, tuple, pd.DataFrame, pd.Index, pd.Series, np.ndarray, pd.RangeIndex, pd.MultiIndex])
    if   type(target) == list or type(target) == tuple:
        return np.array(target)
    elif type(target) == pd.DataFrame or type(target) == pd.Index or type(target) == pd.Series or type(target) == pd.RangeIndex or type(target) == pd.MultiIndex:
        return target.values
    else:
        # np.ndarrayのみを想定
        return target

def conv_pd_index(target) -> pd.Index:
    check_type(target, [list, tuple, pd.Series, np.ndarray, pd.Index, pd.RangeIndex, pd.MultiIndex])
    if   type(target) == list or type(target) == tuple or type(target) == pd.Series or type(target) == np.ndarray:
        return pd.Index(target)
    else:
        # np.ndarrayのみを想定
        return target

def to_string_all_columns(df: pd.DataFrame, n_round=3, rep_nan: str="%%null%%", strtmp: str="-9999999") -> pd.DataFrame:
    """
    画面に表示するときなど、本来intの値は整数で表示したい。
    しかし、nan が含まれているとfloat 型のカラムになるため、整数の表示が崩れる。
    なので、全てのnanを変換して、intのものは整数になるように調整する
    Params::
        df: input data frame
        n_round: float の場合、何桁まで丸めるか
        rep_nan: 変換後の欠損値を表す文字列
        strtmp: 欠損値を一時的に変換する文字列. 置き換える値がdfの中にあるとerror
    """
    df = df.copy()
    # はじめにfloatはraundしておく( == np.float や float では一気にfloatを抜き出せなかった)
    for x in df.columns[df.dtypes == np.float64]: df[x] = df[x].round(n_round)
    for x in df.columns[df.dtypes == np.float32]: df[x] = df[x].round(n_round)
    for x in df.columns[df.dtypes == np.float16]: df[x] = df[x].round(n_round)
    # boolean は予めint に変換しておいて、copy時は0, 1, null で入力したい
    for x in df.columns[df.dtypes == bool]: df[x] = df[x].astype(np.int8)
    df = df.fillna(rep_nan).astype(str).copy()
    if (df == strtmp).sum(axis=0).sum() > 0:
        raise Exception(f"strtmp: {strtmp} exist.")
    def __work(se: pd.Series, except_strings: List[str]):
        if   is_integer_str_regex(se, except_strings=except_strings).sum() == se.shape[0]:
            return se.replace(rep_nan, strtmp,inplace=False).astype(np.float64).astype(np.int64).astype(str).replace(strtmp, rep_nan)
        elif is_float_str_regex(  se, except_strings=except_strings).sum() == se.shape[0]:
            return se.replace(rep_nan, np.nan,inplace=False).astype(np.float32).round(n_round).astype(str).replace(str(np.nan), rep_nan)
        else:
            return se
    df = df.apply(lambda x: __work(x, [rep_nan]), axis=0)
    return df

def is_integer_str_regex(se: pd.Series, except_strings: List[str] = [""]) -> pd.Series:
    se_bool = (se.str.contains(r"^[0-9]$")     | se.str.contains(r"^-[0-9]+$") | \
               se.str.contains(r"^[0-9]\.0+$") | se.str.contains(r"^-[0-9]+\.0+$") | \
               se.str.contains(r"^[1-9][0-9]+$")     | se.str.contains(r"^-[1-9][0-9]+$") | \
               se.str.contains(r"^[1-9][0-9]+\.0+$") | se.str.contains(r"^-[1-9][0-9]+\.0+$"))
    # int64 以上は integer としない
    se_bool = se_bool & (se.str.zfill(len("9223372036854775807")) <= "9223372036854775807")
    for x in except_strings: se_bool = se_bool | (se == x)
    return se_bool

def is_float_str_regex(se: pd.Series, except_strings: List[str] = [""]) -> pd.Series:
    se_bool = (se.str.contains(r"^[0-9]\.[0-9]+$")       | se.str.contains(r"^-[0-9]\.[0-9]+$") | \
               se.str.contains(r"^[1-9][0-9]+\.[0-9]+$") | se.str.contains(r"^-[1-9][0-9]+\.[0-9]+$"))
    for x in except_strings: se_bool = se_bool | (se == x)
    return se_bool

def is_numeric_str_regex(se: pd.Series, except_strings: List[str] = [""]) -> pd.Series:
    se_bool = (se.str.contains(r"^[0-9]$"        )       | se.str.contains(r"^-[0-9]$"        ) | \
               se.str.contains(r"^[0-9]\.[0-9]+$")       | se.str.contains(r"^-[0-9]\.[0-9]+$") |        
               se.str.contains(r"^[1-9][0-9]+$"        ) | se.str.contains(r"^-[1-9][0-9]+$"        ) | \
               se.str.contains(r"^[1-9][0-9]+\.[0-9]+$") | se.str.contains(r"^-[1-9][0-9]+\.[0-9]+$"))
    for x in except_strings: se_bool = se_bool | (se == x)
    return se_bool

def is_regex_pattern(val, ndf: list, regstr: str=r"^%X%$") -> np.ndarray:
    """
    columns の中に、ndf と regstr を使った正規表現に当てはまるかどうかをbooleで返却する
    Return::
        np.array([False, True, ...])
    Params::
        val: dataframe の index or columns or series. str が呼び出せるもの
        ndf: 発見したい文字列のリスト
        regstr: 正規表現のパターン. %X% には ndf の値が代入される
    """
    bool_ret = np.array([False] * val.shape[0])
    for x in ndf: bool_ret = bool_ret | (val.str.contains(regstr.replace(r"%X%",x),regex=True))
    return bool_ret

def index_regex_pattern(val, ndf: list, regstr: str=r"^%X%$") -> np.ndarray:
    """
    columns の中に、ndf と regstr を使った正規表現に当てはまったindexを返却する
    Return::
        np.array([A, B, ...])
    Params::
        val: dataframe の index or columns or series. str が呼び出せるもの
        ndf: 発見したい文字列のリスト
        regstr: 正規表現のパターン. %X% には ndf の値が代入される
    """
    boolwk = is_regex_pattern(val, ndf, regstr=regstr)
    return conv_ndarray(val)[boolwk]

def is_column_string(se: pd.Series) -> bool:
    """ dataframe では str と object の区別がつかない """
    if se.apply(lambda x: type(x) == str).sum() == se.shape[0]:
        return True
    else:
        return False

def is_columns_string(df: pd.DataFrame) -> np.ndarray:
    """
    Return::
        それぞれのカラムがstring type かどうかのlistを返却する
        np.array([False, True, ...])
    """
    return df.apply(lambda se: is_column_string(se), axis=0).values

def drop_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    database から取得すると、カラム名が同じでもdataframeに格納されている
    重複したカラムを落として返却する
    """
    list_bool, listwk = [], []
    for x in df.columns:
        if (df.columns == x).sum() == 1:
            list_bool.append(True)
            listwk.append(x)
        else:
            ## カラムが2個以上の場合
            if x in listwk:
                ## １個目がカウントされていればfalse
                list_bool.append(False)
            else:
                list_bool.append(True)
                listwk.append(x)
    return df.loc[:, list_bool]

def to_pickles(*args, dict_locals=locals(), outdir="./"):
    """
    引数に入力したdataframeをpickle化する
    dict_locals=locals() これは呼び出し側で必ず定義する
    """
    outdir = correct_dirpath(outdir)
    for df in args:
        for val_name in dict_locals.keys():
            if id(df) == id(dict_locals[val_name]):
                df.to_pickle(outdir + val_name + ".pickle")

def check_dataframe_diff(df1: pd.DataFrame, df2: pd.DataFrame, logger: logging.Logger = None, max_count: int=10, ignore_same: bool=False):
    if logger is None:
        from kkpackage.util.logger import set_logger # 常に使いたくないのでここで呼ぶ
        logger = set_logger(__name__)
    df1, df2 = df1.copy().fillna(0), df2.copy().fillna(0) # Series内の nan == nan は False になるので fill しておく

    logger.info("check dataframe shape.", color=["BOLD", "GREEN"])
    logger.info(f"df1 shape: {df1.shape}")
    logger.info(f"df2 shape: {df2.shape}")

    logger.info("check dataframe index.", color=["BOLD", "GREEN"])
    ndf1, ndf2 = df1.index, df2.index
    if (ndf1.shape[0] != ndf2.shape[0]) or (~(ndf1 == ndf2).sum() > 0):
        logger.warning(f"index is different.")
        same_index = values_include(ndf1, ndf2)
        logger.debug(f"same index: {same_index}")
        logger.warning(f"only df1 index: {values_not_include(ndf2, ndf1)}")
        logger.warning(f"only df2 index: {values_not_include(ndf1, ndf2)}")
    else:
        if ignore_same == False:
            logger.info(f"index is same.", color=["BOLD", "BLUE"])

    logger.info("check dataframe columns.", color=["BOLD", "GREEN"])
    ndf1, ndf2 = df1.columns, df2.columns
    same_columns = values_include(ndf1, ndf2)
    if (ndf1.shape[0] != ndf2.shape[0]) or (~(ndf1 == ndf2).sum() > 0):
        logger.warning(f"columns is different.")
        logger.debug(f"same columns: {same_columns}")
        logger.warning(f"only df1 index: {values_not_include(ndf2, ndf1)}")
        logger.warning(f"only df2 index: {values_not_include(ndf1, ndf2)}")
    else:
        if ignore_same == False:
            logger.info(f"columns is same.", color=["BOLD", "BLUE"])

    logger.info("we check only same indexes and same columns", color=["BOLD", "GREEN"])
    df1 = df1.loc[df1.index.isin(df2.index), df1.columns.isin(df2.columns)]
    df2 = df2.loc[df1.index, df1.columns]

    logger.info("check whole data.", color=["BOLD", "GREEN"])
    for x in same_columns:
        sebool = (df1[x] == df2[x])
        if (~sebool).sum() > 0:
            logger.warning(f'"{x}" is different. different values: {[(_x, _y, ) for _x, _y in zip(df1.loc[~sebool, x].iloc[:max_count].values, df2.loc[~sebool, x].iloc[:max_count].values)]}')
        else:
            if ignore_same == False:
                logger.info(f'"{x}" is same.', color=["BOLD", "BLUE"])

def sort_columns_based_first_character(columns: pd.Index, list_order: List[str]) -> np.ndarray:
    """
    columns の 先頭文字をlist_order の順番に沿って並び替える
    Params::
        columns: 対象のcolumn
        list_order: 先頭文字の順番
    """
    columns = columns.copy()
    df_col = pd.DataFrame(columns.values, columns=["colname"])
    df_col["order"] = np.nan # None で初期化する. ここに順番を表す数字を入力する
    # 先頭文字が長い順にsortして、長い方から当てはめていく
    df = pd.DataFrame(list_order, columns=["first_character"])
    df["length"] = df["first_character"].apply(lambda x: len(x))
    df = df.sort_values(by=["length"], ascending=False)
    for i in df.index:
        first_character = df.loc[i, "first_character"]
        boolwk = (df_col["colname"].str.contains("^"+first_character) & df_col["order"].isna())
        df_col.loc[boolwk, "order"] = i
    val = df_col["order"].fillna(-1).max() # dtype=object で 1, nan, None の列のmax()は空になる
    df_col["order"] = df_col["order"].fillna(val+1)
    
    return df_col.sort_values(by=["order", "colname"])["colname"].values

def nanmap(columns: pd.Index, dict_map: dict) -> np.ndarray:
    """
    columns.map(dict) では dict に定義されていない変数は nan になる. 
    定義されてないcolumnはそのまま残すようにする
    """
    columns   = columns.copy()
    columnswk = columns.map(dict_map).fillna("__work").values
    columns   = columns.values
    columns[(columnswk != "__work")] = columnswk[(columnswk != "__work")]
    return columns
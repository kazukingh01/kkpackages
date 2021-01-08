import psycopg2
import re
from typing import List
import pandas as pd

# local package
from kkutils.util.dataframe import values_not_include, to_string_all_columns, drop_duplicate_columns
from kkutils.util.com import strfind, check_type, set_logger
_logname = __name__


__all__ = [
    "Psgre",
]


class Psgre:
    """
    一連のtransactionを同一のinstanceで実現することと
    DataFrameのinterfaceを実装する
    Note::
        connection_string = None のときは空更新を可能にする
    """
    def __init__(self, connection_string: str, max_disp_len: int=100, log_level="info", logfilepath: str=None):
        """
        postgresql db との connection を確立する
        Params::
            connection_string: 接続文字列
                ex) host=172.18.10.2 port=5432 dbname=boatrace user=postgres password=postgres
        """
        self.con = None if connection_string is None else psycopg2.connect(connection_string)
        self.max_disp_len = max_disp_len
        self.sql_list = [] # insert, update, delete は一連のsqlをsetした後に一気に実行することにする
        self.logger = set_logger(_logname+".Psgre."+str(id(self.con)), log_level=log_level, internal_log=False, logfilepath=logfilepath)
        if connection_string is None:
            self.logger.info("dummy connection is established.")
        else:
            self.logger.info(f'connection is established. {connection_string[:connection_string.find("password")]}')


    def __del__(self):
        if self.con is not None:
            self.con.close()


    def __check(self, check_list: List[str] = ["open"]):
        if self.con is not None:
            if "open" in check_list and self.con.closed == 1:
                self.raise_error("connection is closed.")
            if "lock" in check_list and len(self.sql_list) > 0:
                self.raise_error("sql_list is not empty. you can do after ExecuteSQL().")
            if "esql" in check_list and len(self.sql_list) == 0:
                self.raise_error("sql_list is empty. you set executable sql.")


    def __display_sql(self, sql: str) -> str:
        return ("SQL:" + sql[:self.max_disp_len] + " ..." + sql[-self.max_disp_len:] if len(sql) > self.max_disp_len*2 else sql)


    # postgres の raise_error は connection も切るため独自に定義する
    def raise_error(self, msg: str, exception = Exception):
        self.__del__()
        self.logger.raise_error(msg, exception)


    def select_sql(self, sql: str) -> pd.DataFrame:
        self.logger.debug("START")
        self.__check(["open","lock"])
        df = pd.DataFrame()
        if strfind(r"^select", sql, flags=re.IGNORECASE):
            self.logger.debug(self.__display_sql(sql))
            if self.con is not None:
                self.con.autocommit = True # 参照でも原則ロックがかかるため自動コミットONにしておく
                df = pd.read_sql_query(sql, self.con)
                self.con.autocommit = False
        else:
            self.raise_error(f"sql: {sql[:100]}... is not started 'SELECT'")
        df = drop_duplicate_columns(df)
        self.logger.debug("END")
        return df


    def set_sql(self, sql: List[str]):
        self.logger.debug("START")
        check_type(sql, [str, list])
        if type(sql) == str:
            x = sql
            if strfind(r"^select", x, flags=re.IGNORECASE):
                self.raise_error(self.__display_sql(x) + ". you can't set 'SELECT' sql.")
            else:
                self.sql_list.append(x)
        elif type(sql) == list:
            for x in sql:
                if strfind(r"^select", x, flags=re.IGNORECASE):
                    self.raise_error(self.__display_sql(x) + ". you can't set 'SELECT' sql.")
                else:
                    self.sql_list.append(x)
        self.logger.debug("END")


    def execute_sql(self, sql=None):
        """
        sql_list の中身を実行する. sqlがNoneでなければsql_listにsetして実行する
        """
        self.logger.debug("START")
        self.__check(["open"])
        if sql is not None:
            self.__check(["lock"]) # sql がある場合は現在sqlが溜まっていない事を要求する
            self.set_sql(sql)
        self.__check(["esql"])

        if self.con is not None:
            self.con.autocommit = False # 自動コミットをOFFに.
            cur = self.con.cursor()
            try:
                for x in self.sql_list:
                    self.logger.info(self.__display_sql(x))
                    cur.execute(x)
                self.con.commit()
            except:
                self.con.rollback()
                cur.close()
                self.raise_error("sql error !!")
            cur.close()
        self.sql_list = [] # 初期化しておく
        self.logger.debug("END")


    def execute_copy_from_df(
            self, df: pd.DataFrame, tblname: str, system_colname_list: List[str] = ["sys_updated"], 
            filename: str="./postgresql_copy.csv", encoding: str="utf8", n_round: int=3, 
            str_null :str="%%null%%", check_columns: bool=True
    ):
        """
        Params::
            encoding: "shift-jisx0213", "utf8", ...
        """
        self.logger.debug("START")
        self.__check(["open", "lock"])

        # まずテーブルのカラムを取ってくる
        dfcol = self.read_table_layout(tblname, system_colname_list=system_colname_list)
        if check_columns:
            ndf = values_not_include(df.columns, dfcol["colname"].values)
            if ndf.shape[0] > 0:
                self.raise_error(f"{ndf} columns is not included in df: {df.columns}.")
        else:
            df = df.copy()
            for x in dfcol["colname"].values:
                if (df.columns == x).sum() == 0:
                    df[x] = float("nan")
        df = df[dfcol["colname"].values].copy()
        # df を文字列に返還
        df = to_string_all_columns(df, n_round=n_round, rep_nan=str_null, strtmp="-9999999")
        # 文字列のうち、改行やタブをスペースに変換する
        df = df.replace("\r\n", " ").replace("\n", " ").replace("\t", " ")
        df.to_csv(filename, encoding=encoding, quotechar="'", sep="\t", index=False, header=False)
        if self.con is not None:
            try:
                ### カーソルの定義（結果を出力するのに使用する）
                cur = self.con.cursor()
                ### COPY の実行
                f = open(filename, mode="r", encoding=encoding)
                cur.copy_from(f, tblname, columns=tuple(df.columns.tolist()), sep="\t", null=str_null)
                self.con.commit() # これ必要か不明
                self.logger.info("finish to copy from csv.")
            except:
                self.con.rollback() # これ必要か不明
                cur.close()
                self.raise_error("csv copy error !!")

        self.logger.debug("END")
        return df


    def read_table_layout(self, tblname: str, system_colname_list: List[str] = ["sys_updated"]) -> pd.DataFrame:
        self.logger.debug("START")
        sql = "SELECT table_name as tblname, column_name as colname FROM information_schema.columns where table_schema = 'public' and table_name = '" + tblname + "' "
        for x in system_colname_list: sql += "and column_name <> '" + x + "' "
        sql += "order by ordinal_position;"
        df = self.select_sql(sql)
        self.logger.debug("END")
        return df


    def insert_from_df(self, df: pd.DataFrame, tblname: str, set_sql: bool=True, n_round: int=3, str_null :str="%%null%%"):
        sql = "insert into "+tblname+" ("+",".join(df.columns.tolist())+") values "
        df = to_string_all_columns(df, n_round=n_round, rep_nan=str_null, strtmp="-9999999") # 全て文字列に置き換える
        for ndf in df.values:
            sql += "('" + "','".join(ndf.tolist()) + "'), "
        sql = sql[:-2] + "; "
        sql = sql.replace("'"+str_null+"'", "null")
        if set_sql: self.set_sql(sql)
        return sql

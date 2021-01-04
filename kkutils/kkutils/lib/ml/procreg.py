import numpy as np
import pandas as pd
from typing import List

# local package
from kkutils.lib.ml.procs import MyAsType
from kkutils.util.com import check_type, is_callable, set_logger
logger = set_logger(__name__)


__all__ = [
    "ProcRegistry",
]


class ProcRegistry(object):
    def __init__(self, colname_explain: np.ndarray, colname_answer: np.ndarray):
        super().__init__()
        self.processing = {}
        self.default_proc(colname_explain, colname_answer)


    def default_proc(self, colname_explain: np.ndarray, colname_answer: np.ndarray):
        logger.info("START")
        check_type(colname_explain, [np.ndarray])
        check_type(colname_answer,  [np.ndarray])
        self.processing["default_x"] = {}
        self.processing["default_x"]["type"] = "x"
        self.processing["default_x"]["cols"] = colname_explain
        self.processing["default_x"]["proc"] = []
        self.processing["default_y"] = {}
        self.processing["default_y"]["type"] = "y"
        self.processing["default_y"]["cols"] = colname_answer
        self.processing["default_y"]["proc"] = []
        self.processing["default_row"] = {}
        self.processing["default_row"]["type"] = "row"
        self.processing["default_row"]["cols"] = None
        self.processing["default_row"]["proc"] = []
        logger.info("END")
    

    def __call__(self, df: pd.DataFrame, autofix: bool=False, x_proc: bool=True, y_proc: bool=True, row_proc: bool=True):
        logger.info("START")
        # row proc
        if row_proc:
            df = self.proc_row(df)
            if x_proc == False and y_proc == False:
                ## df だけの返却方法
                return df
        # col proc
        list_x, list_y = [], []
        for name in self.processing.keys():
            if self.processing[name]["type"] not in ["x", "y"]: continue
            if x_proc == False and self.processing[name]["type"] == "x": continue
            if y_proc == False and self.processing[name]["type"] == "y": continue
            logger.info(f'name: {name}, type: {self.processing[name]["type"]}')
            if len(self.processing[name]["proc"]) > 0 and isinstance(self.processing[name]["proc"][0], MyAsType):
                ## 始めの処理がAsTypeの場合はメモリの軽減のためにndf前に適応する
                ndf = df[self.processing[name]["cols"]].astype(self.processing[name]["proc"][0].convert_type).values.copy()
            else:
                ndf = df[self.processing[name]["cols"]].values.copy()
            for _proc in self.processing[name]["proc"]:
                logger.info(f'proc: {_proc}')
                shape_before = ndf.shape
                logger.info(f"before shape: {ndf.shape}")
                ndf = _proc(ndf)
                logger.info(f"after  shape: {ndf.shape}")
                if shape_before[0] != ndf.shape[0]:
                    logger.warning("The number of rows is different from before and after process.")
            if   self.processing[name]["type"] == "x": list_x.append(ndf)
            elif self.processing[name]["type"] == "y": list_y.append(ndf)
        if autofix:
            if len(list_x) == 1: list_x = list_x[0]
            if len(list_y) == 1: list_y = list_y[0]
        logger.info(f"after processing x: \n{list_x}")
        logger.info(f"after processing y: \n{list_y}")
        logger.info("END")
        return list_x, list_y
    

    def proc_row(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("START")
        df = df.copy()
        for name in self.processing.keys():
            if self.processing[name]["type"] not in ["row"]: continue
            logger.info(f'name: {name}, type: {self.processing[name]["type"]}')
            for _proc in self.processing[name]["proc"]:
                logger.info(f'proc: {_proc}')
                logger.info(f"before shape: {df.shape}")
                df = _proc(df)
                logger.info(f"after  shape: {df.shape}")
        logger.info("END")
        return df


    def register(self, list_proc: list, name: str=None, type_proc: str=None, columns: np.ndarray=None):
        """
        処理を登録する. 登録名が新規の場合は新たにprocessingを登録する
        Params::
            list_proc: callable な関数やクラスのリスト
            name: 登録処理名
            type_proc: "x" or "y". rowの場合、default_rowに追加するのみとする
        """
        logger.info("START")
        if name is None and isinstance(type_proc, str) and type_proc == "x":   name = "default_x"
        if name is None and isinstance(type_proc, str) and type_proc == "y":   name = "default_y"
        if name is None and isinstance(type_proc, str) and type_proc == "row": name = "default_row"
        if name not in list(self.processing.keys()):
            if type_proc not in ["x", "y"]:
                logger.raise_error(f'type_proc must be "x" or "y". type_proc: {type_proc}')
            self.processing[name] = {}
            self.processing[name]["type"] = type_proc
            self.processing[name]["cols"] = columns
            self.processing[name]["proc"] = []
        for _proc in list_proc:
            self.processing[name]["proc"].append(_proc)
        logger.info("END")
    

    def set_columns(self, columns: np.ndarray, name: str=None, type_proc: str=None):
        """
        説明変数を再度セットする
        Params::
            columns: 新規の説明変数
            name: None の場合, type_proc が指定されていれば、defaultに自動セットする
            type_proc: x or y
        """
        if name is None and isinstance(type_proc, str) and type_proc == "x": name = "default_x"
        if name is None and isinstance(type_proc, str) and type_proc == "y": name = "default_y"
        self.processing[name]["cols"] = columns
    

    def fit(self, df: pd.DataFrame):
        """
        登録した処理に関するパラメータを学習するため、入力データを基準にfittingさせる
        """
        logger.info("START")
        df = df.copy()
        for name in self.processing.keys():
            if self.processing[name]["type"] not in ["row"]: continue
            logger.info(f'name: {name}, type: {self.processing[name]["type"]}')
            for _proc in self.processing[name]["proc"]:
                logger.info(f'proc: {_proc}')
                logger.info(f"before shape: {df.shape}")
                if is_callable(_proc, "fit"):
                    # Fitting
                    _proc.fit(df)
                df = _proc(df)
                logger.info(f"after  shape: {df.shape}")
        for name in self.processing.keys():
            if self.processing[name]["type"] not in ["x", "y"]: continue
            logger.info(f'name: {name}, type: {self.processing[name]["type"]}')
            if len(self.processing[name]["proc"]) > 0 and isinstance(self.processing[name]["proc"][0], MyAsType):
                ## 始めの処理がAsTypeの場合はメモリの軽減のためにndf前に適応する
                ndf = df[self.processing[name]["cols"]].astype(self.processing[name]["proc"][0].convert_type).values.copy()
            else:
                ndf = df[self.processing[name]["cols"]].values
            for _proc in self.processing[name]["proc"]:
                logger.info(f'proc: {_proc}')
                if is_callable(_proc, "fit"):
                    # Fitting
                    _proc.fit(ndf)
                logger.info(f"before shape: {ndf.shape}")
                ndf = _proc(ndf) # fit後、適用する
                logger.info(f"after  shape: {ndf.shape}")
        logger.info("END")


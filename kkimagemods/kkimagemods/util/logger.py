import logging, io, os, datetime

# local package
from kkimagemods.util.common import correct_dirpath

# logger name を保管しておく. loggingから名前空間にアクセスできそうにない.
_list_logname  = []
_dict_loglevel = {"info":logging.INFO, "debug":logging.DEBUG, "warn":logging.WARNING, "error":logging.ERROR}
_formatter = logging.Formatter('%(asctime)s - %(name)s - %(funcName)s - %(levelname)s : %(message)s')

class MyLogger(logging.Logger):
    def __init__(self):
        super().__init__(self)
        self.internal_stream = io.StringIO() # この型とわかるように初期化しておく

    def raise_error(self, msg: str, exception = Exception):
        self.error(msg)
        raise exception(msg)

    def set_internal(self):
        """ 内部ログ用 """
        for x in self.handlers:
            if x.get_name == "internal_logger":
                # 既に設定されている場合はそれ以上setしない
                return None
        self.internal_stream = io.StringIO()
        s_handler = logging.StreamHandler(self.internal_stream)
        s_handler.setFormatter(_formatter)
        s_handler.set_name("internal_logger")
        self.addHandler(s_handler)

    def set_logfile(self, filepath: str):
        """
        ログファイル用
        Params::
            filepath: 日付を設定する場合は%YYYY%,%MM%,%DD%の文字列を入れる
        """
        for x in self.handlers:
            if x.get_name == "logfile_logger":
                # 既に設定されている場合はそれ以上setしない
                return None
        str_year  = datetime.datetime.now().strftime("%Y")
        str_month = datetime.datetime.now().strftime("%m")
        str_day   = datetime.datetime.now().strftime("%d")
        filepath  = filepath.replace("%YYYY%",str_year).replace("%MM%",str_month).replace("%DD%",str_day)
        f_handler = logging.FileHandler(filename=filepath)
        f_handler.setFormatter(_formatter)
        f_handler.set_name("logfile_logger")
        self.addHandler(f_handler)


def set_logger(name: str =  __name__, log_level: str = "info", internal_log: bool=False, logfilepath: str=None) -> MyLogger:
    """logging の名前空間にあるアドレスを返却する"""
    global _list_logname
    global _dict_loglevel
    global _formatter

    logger = logging.getLogger(name)
    logger.__class__ = MyLogger # cast
    if name in _list_logname:
        pass
    else:
        ## 標準出力用
        s_handler = logging.StreamHandler()
        s_handler.setFormatter(_formatter)
        logger.addHandler(s_handler)
        ## ログ出力レベルの統一
        logger.setLevel(_dict_loglevel[log_level])
        _list_logname.append(name)
        if internal_log:
            logger.set_internal()
        if logfilepath is not None:
            logger.set_logfile(logfilepath)
    return logger


def set_loglevel(name: str = None, log_level: str = "info"):
    """
    ログレベルを設定する
    Params::
        name: log name. もしNoneなら全ての名前のloggeを変える
        log_level: info, debug, warn, error
    Usage::
        >>> set_loglevel(name=None, log_level="debug")
    """
    global _list_logname
    global _dict_loglevel
    if name is None:
        for x in _list_logname:
            logging.getLogger(x).setLevel(_dict_loglevel[log_level])
    else:
        if name in _list_logname:
            logging.getLogger(name).setLevel(_dict_loglevel[log_level])
        else:
            raise Exception("No name in logging space.")
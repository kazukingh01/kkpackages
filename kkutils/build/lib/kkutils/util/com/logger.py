import logging, io, os, datetime

# local package
from kkutils.util.com import correct_dirpath


__all__ = [
    "set_logger",
    "set_loglevel",
    "MyLogger",
]


# logger name を保管しておく. loggingから名前空間にアクセスできそうにない.
_list_logname  = []
_dict_loglevel = {"info":logging.INFO, "debug":logging.DEBUG, "warn":logging.WARNING, "error":logging.ERROR}
_formatter = logging.Formatter('%(asctime)s - %(name)s - %(funcName)s - %(levelname)s : %(message)s')
_pycolor = {
    "BLACK"     : '\033[30m',
    "RED"       : '\033[31m',
    "GREEN"     : '\033[32m',
    "YELLOW"    : '\033[33m',
    "BLUE"      : '\033[34m',
    "PURPLE"    : '\033[35m',
    "CYAN"      : '\033[36m',
    "WHITE"     : '\033[37m',
    "END"       : '\033[0m',
    "BOLD"      : '\033[1m',
    "UNDERLINE" : '\033[4m',
    "INVISIBLE" : '\033[08m',
    "REVERCE"   : '\033[07m',        
}

class MyFormatter(logging.Formatter):
    """
    Formatter の def format を override して、color code の消し込みを行う
    """
    global _pycolor
    def format(self, record) -> str:
        string = super().format(record)
        for x in _pycolor.values(): string = string.replace(x, "")
        return string
_formatter_outfile = MyFormatter('%(asctime)s - %(name)s - %(funcName)s - %(levelname)s : %(message)s')



class MyLogger(logging.Logger):
    global _pycolor
    def __init__(self):
        super().__init__(self)
        self.internal_stream = io.StringIO() # この型とわかるように初期化しておく
    
    def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False, color: str=None):
        if color is None and level == logging.WARNING: color = ["BOLD", "YELLOW"]
        if color is None and level == logging.ERROR:   color = ["BOLD", "RED"]
        if color is not None: msg = (_pycolor[color] if type(color) == str else "".join([_pycolor[x] for x in color])) + msg + _pycolor["END"]
        super()._log(level, msg, args, exc_info=exc_info, extra=extra, stack_info=stack_info)

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
        f_handler.setFormatter(_formatter_outfile)
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

import logging

def set_looger(name: str =  __name__, log_level: str = "info") -> logging.Logger:
    logger = logging.getLogger(name)
    if len(logger.handlers) == 0:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(funcName)s - %(levelname)s : %(message)s')
        ## 標準出力用
        s_handler = logging.StreamHandler()
        s_handler.setFormatter(formatter)
        logger.addHandler(s_handler)
    ## ログ出力レベルの統一
    logger.setLevel({"info":logging.INFO, "debug":logging.DEBUG, "warn":logging.WARNING, "error":logging.ERROR}[log_level])
    return logger


import time
from urllib import request
import requests
import subprocess
import psutil
from time import sleep
import signal

# local package
from kkpackage.util.logger import set_logger
logger = set_logger(name=__name__)


def download(url: str, savepath: str, proxies: dict=None):
    """
    file を download する
    Params::
        url: download url
        savepath: 保存先のpath
        proxies: 下記のように指定
            proxies = {
                "http": "http://kume.kazuki@jp.fujitsu.com:0879424845@moapgw.gw.fujitsu.co.jp:8080" \
            }
    """
    if proxies is not None:
        proxyHandler = request.ProxyHandler(proxies)
        opener = request.build_opener(proxyHandler)
        request.install_opener(opener)
    logger.info(f"download url: {url}")
    request.urlretrieve(url, savepath)


def get_html(url: str, proxies: dict=None, retry: int=1) -> (str, bytes, ):
    """
    html の文字列を返却する

    Params::
        url: url
        proxies: 次のような使い方
            proxies = {
                'http': 'http://proxy_host:8080',
                'https': 'http://proxy_host:8080',
            }
            socks の場合
            proxies = {
                'http':  'socks5h://localhost:8080',
                'https': 'socks5h://localhost:8080',
            }
    note::
        pip install pysocks が必要
    """
    logger.info(f"url: {url}")
    res, count = None, 0
    while(count < retry):
        try:
            res = requests.get(url, proxies=(proxies if proxies is not None else {}) )
            res.raise_for_status() # 正常に取得できなければエラーをraiseする
            break
        except requests.exceptions.ConnectionError:
            count += 1
            logger.warning(f"connection error. retry: {count}")
            time.sleep(3)
        except Exception:
            # HTTPエラーの場合もループさせる
            count += 1
            logger.warning(f"HTTP error. retry: {count}")
    res.raise_for_status() # 正常に取得できなければ再度エラーをraiseする
    return res.text, res.content


def tor(proc_name: str="tor", restart: bool=False, timeout: int=20) -> bool:
    """
    tor を起動する, timeout を設定できる
    """
    def receive_alarm(signum, stack):
        raise TimeoutError("timeout")
    signal.signal(signal.SIGALRM, receive_alarm)
    signal.alarm(timeout)

    try:
        # process が起動しているかどうか
        for proc in psutil.process_iter():
            if proc.name() == proc_name:
                if restart: proc.kill()
                else:
                    signal.alarm(0)
                    return True
        
        logger.info("tor proccess is starting...")
        proc = subprocess.Popen(proc_name, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        while True:
            stdout = proc.stdout.readline()
            logger.info(stdout)
            if str(stdout).find("Bootstrapped 100%: Done") >= 0:
                signal.alarm(0)
                logger.info("tor proccess is running !!")
                break
            sleep(0.1)
    except TimeoutError:
        logger.error(f"time: {timeout} is over.")
        signal.alarm(0)
        for proc in psutil.process_iter():
            if proc.name() == proc_name: proc.kill()

        return False

    signal.alarm(0)
    return True


def getproc_with_tor(proc, index_bool: bool=0, max_count: int=5, tor_timeout: int=20):
    """
    主にHTMLのGETプロセスをtorと一緒に使う。
    Params::
        proc: get process. partial で埋め込んで使用すると良い. proc の返り値には get が成功したかどうかを含む情報が必要
        index_bool: get が成功したかどうかの情報のindex
    """
    restart, count, is_success, outval = False, 0, False, None
    while count < 5:
        is_tor, count_tor = False, 0
        ## default tor を使用する
        while (is_tor == False and count_tor < max_count):
            is_tor = tor(restart=restart, timeout=tor_timeout)
            count_tor += 1
        if is_tor == False: logger.raise_error("tor process is not working.")
        outval = proc()
        if type(outval) == bool:
            is_success = outval
        elif type(outval) in [list, tuple]:
            is_success = outval[index_bool]
        if is_success:
            restart = False
            break
        else:
            count += 1
            logger.warning(f"proc running is fatal. retry: {count}")
            restart = True # proc が失敗したらtorも再起動してみる
    if is_success == False: logger.raise_error("proc runnning error !!")
    return outval
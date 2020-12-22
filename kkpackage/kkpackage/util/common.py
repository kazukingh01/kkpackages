from typing import List, Tuple
import sys, os, glob, re, shutil, pickle, datetime

def is_callable(class_: object, func_: str, list_index=None) -> bool:
    """ 関数がコールできるかどうかを調べる """
    try:
        if list_index == None: getattr(class_, func_)
        else:                  getattr(class_[list_index], func_)
    except AttributeError: return False
    except IndexError: return False
    except TypeError: return False
    return True

def check_type(target, check_list: List[type]):
    if type(target) not in check_list:
        raise Exception(f"target type is wrong[{type(target)}]. target type require {check_list}.")

def correct_dirpath(dirpath: str) -> str:
    if os.name == "nt":
        return dirpath if dirpath[-1] == "\\" else (dirpath + "\\")
    else:
        return dirpath if dirpath[-1] == "/" else (dirpath + "/")

def get_file_list(dirpath: str, regex_list: List[str] = []) -> List[str]:
    dirpath = correct_dirpath(dirpath)
    file_list_org = glob.glob(dirpath + "**", recursive=True)
    file_list     = []
    for regstr in regex_list:
        file_list += list(filter(lambda x: len(re.findall(regstr, x)) > 0, file_list_org))
    return file_list if len(regex_list) > 0 else file_list_org

def get_dir_list(dirpath: str, regex_list: List[str] = []) -> List[str]:
    dirpath = correct_dirpath(dirpath)
    dirlist_org = os.listdir(dirpath)
    dirlist_org = [correct_dirpath(dirpath + x) for x in dirlist_org]
    dirlist = []
    for regstr in regex_list:
        dirlist += list(filter(lambda x: len(re.findall(regstr, x)) > 0, dirlist_org))
    return dirlist if len(regex_list) > 0 else dirlist_org

def rm_files(dirpath: str, regex_list: List[str]):
    for x in get_file_list(dirpath, regex_list=regex_list):
        os.remove(x)

def makedirs(dirpath: str, exist_ok: bool = False, remake: bool = False):
    dirpath = correct_dirpath(dirpath)
    if remake and os.path.isdir(dirpath): shutil.rmtree(dirpath)
    os.makedirs(dirpath, exist_ok = exist_ok)

def get_args() -> dict:
    dict_ret = {}
    args = sys.argv
    dict_ret["__fname"] = args[0]
    for i, x in enumerate(args):
        if   x[:4] == "----":
            # この引数の後にはLISTで格納する
            dict_ret[x[4:]] = []
            for _x in args[i+1:]:
                if _x[:2] != "--": dict_ret[x[4:]].append(_x)
                else: break
        elif x[:3] == "---":
            dict_ret[x[3:]] = True
        elif x[:2] == "--":
            dict_ret[x[2:]] = args[i+1]
    return dict_ret

def check_args(check_list: List[str], caption: str=None):
    args = get_args()
    for x in check_list:
        if args.get(x) is None:
            print(args)
            raise Exception(f'"{x}" parameter is needed. '+("" if caption is None else caption))

def str_to_datetime(string: str) -> datetime.datetime:
    if   strfind(r"^[0-9]+$", string) and len(string) == 14:
        return datetime.datetime(int(string[0:4]), int(string[4:6]), int(string[6:8]), int(string[8:10]), int(string[10:12]), int(string[12:14]))
    else:
        raise ValueError(f"{string} is not converted to datetime.")

def str_to_date(string: str) -> datetime.datetime:
    if   strfind(r"^[0-9]+$", string) and len(string) == 8:
        return datetime.datetime(int(string[0:4]), int(string[4:6]), int(string[6:8]))
    elif strfind(r"^[0-9][0-9][0-9][0-9]/[0-9][0-9]/[0-9][0-9]$", string):
        strwk = string.split("/")
        return datetime.datetime(int(strwk[0]), int(strwk[1]), int(strwk[2]))
    elif strfind(r"^[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]$", string):
        strwk = string.split("-")
        return datetime.datetime(int(strwk[0]), int(strwk[1]), int(strwk[2]))
    else:
        raise ValueError(f"{string} is not converted to datetime.")

def str_to_time(string: str) -> datetime.datetime:
    """
    datetime.datetimeで返却する日付は2000/01/01 で固定する
    """
    if   strfind(r"^[0-9]+$", string) and len(string) == 4:
        return datetime.datetime(2000, 1, 1, int(string[0:2]), int(string[2:4]), 0)
    elif strfind(r"^[0-9]+$", string) and len(string) == 6:
        return datetime.datetime(2000, 1, 1, int(string[0:2]), int(string[2:4]), int(string[4:6]))
    elif strfind(r"^[0-9]:[0-9][0-9]$", string):
        strwk = string.split(":")
        return datetime.datetime(2000, 1, 1, int(strwk[0]), int(strwk[1]), 0)
    elif strfind(r"^[0-9][0-9]:[0-9][0-9]$", string):
        strwk = string.split(":")
        return datetime.datetime(2000, 1, 1, int(strwk[0]), int(strwk[1]), 0)
    elif strfind(r"^[0-9][0-9]:[0-9][0-9]:[0-9][0-9]$", string):
        strwk = string.split(":")
        return datetime.datetime(2000, 1, 1, int(strwk[0]), int(strwk[1]), int(strwk[2]))
    else:
        raise ValueError(f"{string} is not converted to datetime.")

def args_date(*argv) -> Tuple[datetime.datetime]:
    return tuple([(str_to_date(x) if x is not None else None) for x in argv])

def save_pickle(_obj: object, filepath: str):
    with open(filepath, mode='wb') as f:
        pickle.dump(_obj, f)

def load_pickle(filepath: str) -> object:
    _obj = None
    with open(filepath, mode='rb') as f:
        _obj = pickle.load(f)
    return _obj

def strfind(pattern: str, string: str, flags=0) -> bool:
    if len(re.findall(pattern, string, flags=flags)) > 0:
        return True
    else:
        return False

def conv_str_auto(string):
    if   strfind(r"^[0-9]+$", string) or strfind(r"^-[0-9]+$", string) or strfind(r"^[0-9]+\.0+$", string) or strfind(r"^-[0-9]+\.0+$", string): return int(float(string))
    elif strfind(r"^[0-9]+\.[0-9]+$", string) or strfind(r"^-[0-9]+\.[0-9]+$", string): return float(string)
    elif string == "true":  return True
    elif string == "false": return False
    return string

def basename_url(url: str) -> str:
    return url[url.rfind("/")+1:]

def check_list_depth(target: object, depth: int, check_type: type=list):
    if isinstance(target, check_type):
        for _ in range(depth):
            try:
                target[0]
                target = target[0]
            except IndexError:
                raise Exception(f'target: {target} is not depth: {depth}')
    else:
        raise Exception(f'target: {target} is not type: {check_type}')

def get_filename(path: str):
    return ".".join(os.path.basename(path).split(".")[:-1])

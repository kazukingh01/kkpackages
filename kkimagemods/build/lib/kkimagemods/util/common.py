from typing import List, Tuple
import sys, os, glob, re, shutil

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
    for i, x in enumerate(args):
        if x[:2] == "--":
            dict_ret[x[2:]] = args[i+1]
    return dict_ret


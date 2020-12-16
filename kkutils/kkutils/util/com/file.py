from lhafile import Lhafile
import json
from typing import List

# local package
from kkutils.util.com import correct_dirpath


__all__ = [
    "unpack_lzh",
    "open_json",
]


def unpack_lzh(filepath: str, encoding: str="shift-jisx0213", is_save: bool=False, save_outdir="./") -> List[str]:
    archive = Lhafile(filepath)
    list_str = []
    for fmem in archive.filelist:
        data = archive.read(fmem.filename)
        data = data.decode(encoding=encoding)
        list_str.append(data)
        if is_save:
            with open(correct_dirpath(save_outdir) + fmem.filename, mode="w", encoding=encoding) as f:
                f.write(data)
    return list_str

def open_json(filepath: str, encoding: str="utf8") -> dict:
    dict_json = None
    with open(filepath, mode="r", encoding=encoding) as f:
        dict_json = json.load(f)
    return dict_json
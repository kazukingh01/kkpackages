import numpy as np
import cv2
import os
from typing import List, Tuple

# logger の 定義
from kkimagemods.util.logger import set_looger
logger = set_looger(__name__)

# 自作utilのimport
from kkimagemods.util.common import check_type, correct_dirpath

class BaseStreamer:
    def __init__(self):
        self.cap    = cv2.VideoCapture()
    
    def __del__(self):
        self.cap.release()
        logger.info("streamer is released")

    def close(self):
        self.__del__()
    
    def shape(self) -> (int, int):
        """ return (height, width) """
        width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (height, width)
    
    def get_fps(self) -> float:
        return self.cap.get(cv2.CAP_PROP_FPS)

    def is_open(self) -> bool:
        return self.cap.isOpened()

    def get_frame_count(self) -> int:
        if type(self.cap) is str:
            return int(self.cap.get(7))
        else:
            return -1



class Streamer(BaseStreamer):
    def __init__(self, src):
        super().__init__()
        check_type(src, [int, str]) # void function. 型をチェックする
        self.src        = src
        self.is_next    = True
        self.cap        = cv2.VideoCapture(src)
        self.frame_cnt  = 0
        logger.info(f"{self.__class__.__name__} is created")
    
    def get_frame(self) -> (bool, np.ndarray):
        """
        cap の frame を読み取って、nextグラグとframeを返却します。
        """
        logger.debug("START")

        # frame を読み込む
        is_next, frame = self.cap.read()
        logger.debug("reading original frame.")

        # 動画の読み込みに関して、is_next が Falseになれば動画が終了する
        self.is_next = is_next
        self.frame_cnt += 1

        logger.debug("END")
        return is_next, frame
    
    def play(self, open_window: bool = True):
        while self.is_open():
            print(f'Frame: {self.get_frame_count()}, fps: {self.get_fps()}')
            _, frame = self.get_frame()
            if frame is None: break
            print(frame)
            if open_window:
                cv2.imshow('__window', frame)
                # ESCキー押下で終了
                if cv2.waitKey(30) & 0xff == 27:
                    break
        self.close()
    
    def save_images(self, outdir: str, step: int = 1, max_images=10000):
        outdir = correct_dirpath(outdir)
        name_base = os.path.basename(self.src)
        i = 0
        while self.is_open():
            _, frame = self.get_frame()
            if frame is None: break
            if self.frame_cnt % step == 0:
                if i > max_images:
                    raise Exception(f"max images: {max_images}.  save images is too much !!")
                filename = outdir + name_base + "." + str(i).zfill(len(str(max_images))) + ".png"
                logger.info(f"save image: {filename}")
                cv2.imwrite(filename, frame)
                i += 1
        self.close()



class StreamBuffer:
    def __init__(self, max_frames: int = 2):
        self.max_frames = max_frames
        self.frames     = {"org":[None] * self.max_frames} # {org:[frame0, frame1, ...], proc1:[frame0, frame1, ...], ...}
        self.proc_list  = []

    def regist_image_processing(self, name, func, input_frames, *args, **kwargs):
        """
        画像処理を登録させる
        """
        self.proc_list.append({"name":name, "func":func, "input_frames":input_frames, 
                               "args":args, "kwargs":kwargs})
        self.frames[name] = [None] * self.max_frames
        logger.info(f"Regist image processing. name:{name}, func:{func.__name__}," + \
                    f"input_frames:{input_frames}, args:{args}, kwargs:{kwargs}")
    
    def run_proc(self, frame: np.ndarray):
        logger.debug("START")

        # max_frames の数だけ動画を保存する
        ## 初期値としてmax_framesの数だけNoneで埋めているいので、最初から下記処理を実行する
        for x in self.frames.keys():
            self.frames[x] = self.frames[x][0:self.max_frames-1]

        # 動画を保存する
        self.frames["org"].insert(0, frame)

        # 登録されたproc があれば実行する
        for i_proc, dictwk in enumerate(self.proc_list):
            ## proc の情報を読み出し
            name         = dictwk["name"]
            func         = dictwk["func"]
            input_frames = dictwk["input_frames"]
            args         = dictwk["args"]
            kwargs       = dictwk["kwargs"]
            logger.debug(f"image processing {i_proc}. name:[{name}]")

            ## input_frames を作成する。0 が最新のフレーム. 1 が直前のフレームとなる
            frames = [self.frames[_name][_i] for _name, _i in input_frames]
            ## frameがNoneの場合は直前のframeをcopyする
            for i_frame, _frame in enumerate(frames):
                if _frame is None: frames[i_frame] = frames[i_frame - 1].copy()
            frame  = func(*frames, *args, **kwargs)
            self.frames[name].insert(0, frame)

            logger.debug(f"\n{frame}")
        
        logger.debug("END")



class Recorder(BaseStreamer):
    def __init__(self, out_filename: str, fps: float=None, width:int = None, height: int = None, \
                 streamer: BaseStreamer=None, \
                 fourcc: int = cv2.VideoWriter_fourcc(*'XVID')):

        super().__init__()
        check_type(out_filename, [str]) # void function. 型をチェックする

        # streamer があれば関数から自動で取得する
        if streamer is not None:
            self.cap = cv2.VideoWriter(out_filename, fourcc, streamer.get_fps(), streamer.shape()[::-1])
        else:
            self.cap = cv2.VideoWriter(out_filename, fourcc, fps, (width, height))
    
    def write(self, frame: np.ndarray):
        logger.debug("START")
        self.cap.write(frame)
        logger.debug("END")

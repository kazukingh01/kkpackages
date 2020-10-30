# -*- coding: utf-8 -*-

"""
kkimagemods.lib.coco
~~~~~~~~~~~~~~~~~
"""

import pandas as pd
import numpy as np
import cv2
import json, datetime, glob, os, shutil, sys
from typing import List

# my package
from kkimagemods.util.images import drow_bboxes
from kkimagemods.util.common import correct_dirpath, check_type, makedirs, get_file_list, get_filename
from kkimagemods.util.logger import set_logger
logger = set_logger(__name__)

def coco_info(
        description: str = "my coco dataset.",
        url: str = "http://test",
        version: str = "1.0",
        year: str = datetime.datetime.now().strftime("%Y"), 
        contributor: str = "Test",
        date_created: str = datetime.datetime.now().strftime("%Y/%m/%d")
        ):

    _dict = {}
    _dict["description"] = description
    _dict["url"] = url
    _dict["version"] = version
    _dict["year"] = year
    _dict["contributor"] = contributor
    _dict["date_created"] = date_created

    return _dict


class Ndds2Coco:
    """
    NDDS output convert to Coco format
    https://cocodataset.org/#format-data
    """

    def __init__(
        self,
        indir: str,
        ignore_jsons: List[str] = ["_camera_settings.json", "_object_settings.json"],
        setting_json_fname: str = "_object_settings.json",
    ):
        """
        Parmas::
            info: 'info' key of coco format.
            images: Data Farme converted from NDDS outputs
            categories: dictionary for classification of objects named unique tag
        """
        self.info = coco_info()
        self.images  = pd.DataFrame()
        self.df_ndds = pd.DataFrame()
        self.instances = {}
        self.indir = correct_dirpath(indir)
        self.ignore_jsons = ignore_jsons
        self.setting_json_fname = setting_json_fname
        self.convert_mode = None
        self.instance_merge = None
        self.keypoints = None
        self.skelton = None
        logger.debug("Set instance: Ndds2Coco.")
    

    def __get_json_files(self, dirpath: str) -> List[str]:
        """
        Get json files at NDDS output directory. Ignore json files written at 'self.ignore_jsons' (variable).
        """
        logger.debug("START")
        _dirpath = correct_dirpath(dirpath)
        json_files = []
        for x in glob.glob(_dirpath + "*.json"):
            if sum([x.find(y) >= 0 for y in self.ignore_jsons]) > 0: continue
            logger.debug(f"json file: {x}")
            json_files.append(x)

        logger.debug("END")
        return json_files
    

    @classmethod
    def ndds_instanceid_to_color(cls, instance_id: int):
        RGBint = instance_id
        pixel_b =  RGBint & 255
        pixel_g = (RGBint >> 8) & 255
        pixel_r =   (RGBint >> 16) & 255
        return (pixel_b,pixel_g,pixel_r,)


    def preview(self, img_name: str):
        """
        NDDS の image を NDDS json に従って preview する.
        """
        json_path = self.indir + img_name.replace(".png", ".json")
        fjson = json.load(open(json_path))
        img   = cv2.imread(self.indir + img_name)
        bbox  = np.array([[x["bounding_box"]["top_left"][::-1], x["bounding_box"]["bottom_right"][::-1]] for x in fjson["objects"]])
        img   = drow_bboxes(img, bboxes=bbox.reshape(-1, 4).tolist(), bboxes_class=[x["class"] for x in fjson["objects"]], bbox_type="xy")
        cv2.imshow("preview", img)
        cv2.waitKey(0)


    def set_base_parameter(
        self, dirpath: str, n_files: int = 100, 
        instance_merge: dict = None, bboxes: dict=None,
        keypoints: dict = None, keypoints_merge: dict = None,
        segmentations: dict = None,
        visibility_threshold = 0.2, extra_pixel_x: int = 0, extra_pixel_y: int = 0, 
        convert_mode: str = "is"
    ):
        """
        ## Precondition ###########################
        ## All NDDS object's tag named unique !!
        ###########################################
        At first, we need to set gray scale color parameters at *.cs.png of NDDS output files in order to segment object.
        The color parameters is set automatically.
        We sample color of object's center point ('projected_cuboid_centroid' in json file) in *.cs.png images.
        And we divide objects (named unique tag) thier color of .cs.png.
        ※ memo ※
        ※ 最初 instance (.is.png) で判断しようとしたが、複数のobjectがある場合に、別のインスタンスなのに
        ※ 同じ色が採用されているケースがあったので、cs を使用することにした。この場合、class名(tag)は全て別にする必要がある
        
        params::
            dirpath: NDDS output directory path
            n_files: The number of files we sample. (default 100 images)
            instance_merge: If it is not None, objects named unique tag separate instance.(ex. in case that human_a is composed some objects(body, clothes, shoes, ...))
            visibility_threshold: When we sample images, we ignore under visibility_threshold.
        
        Usage::
            >>> ndds_to_coco = Ndds2Coco()
            >>> ndds_to_coco.set_base_parameter(
                    "~/my_ndds_output_path", n_files=100, 
                    instance_merge={
                        "human1":["body1","shoes1","clothes1", ...], 
                        "human2":["body2","shoes2","clothes2", ...], 
                        "dog1":["dog_body1", "dog_collars1"]
                    }, 
                    bboxes={
                        "human1":"bbox_human"
                    },
                    keypoints={
                        "hook":{
                            "kpt_a"    :[{"name":"kpt_a", "type":"center"}],
                            "kpt_b"    :[{"name":"kpt_b", "type":"center"}],
                            "kpt_cb"   :[{"name":"kpt_cb","type":"center"}],
                            "kpt_c"    :[{"name":"kpt_c", "type":"center"}],
                            "kpt_cd"   :[{"name":"kpt_cd","type":"center"}],
                            "kpt_d"    :[{"name":"kpt_d", "type":"center"}],
                            "kpt_e"    :[{"name":"kpt_e", "type":"center"}],
                            "kpt_dl_dr":[{"name":"kpt_dl","type":"left"  }, {"name":"kpt_dr","type":"right" }],
                        },
                    },
                    segmentation={
                        "hook":{
                            "name": "bbox_human", "type": "inbox"
                        }
                    },
                    visibility_threshold = 0.2
                    )
        """
        logger.debug("START")
        self.convert_mode   = convert_mode
        self.instance_merge = instance_merge
        self.bboxes = bboxes
        self.keypoints = keypoints
        self.keypoints_merge = keypoints_merge
        self.segmentations = segmentations
        if self.convert_mode == "cs":
            df = pd.DataFrame()

            # json files
            files = self.__get_json_files(dirpath)
            count_file = 0
            list_df = []
            for i in np.random.permutation(np.arange(len(files))):
                # json file read
                fjson = json.load(open(files[i]))
                logger.info(f"open file: {files[i]}")

                # image file read
                img = cv2.cvtColor(cv2.imread(files[i].replace(".json", ".cs.png")), cv2.COLOR_BGR2GRAY)

                # json file reading
                for dictwk in fjson["objects"]:
                    if dictwk["visibility"] < visibility_threshold: continue
                    x = dictwk["projected_cuboid_centroid"][0]
                    y = dictwk["projected_cuboid_centroid"][1]
                    logger.debug(f'fname: {os.path.basename(files[i])}, class: {dictwk["class"]}, visibility: {dictwk["visibility"]}, center: {(int(y), int(x), )}')
                    listwk = None
                    x1, x2  = (int(x) + -1 * extra_pixel_x), (int(x) + extra_pixel_x+1)
                    y1, y2  = (int(y) + -1 * extra_pixel_y), (int(y) + extra_pixel_y+1)
                    x1 = x1 if x1 >= 0 else 0
                    y1 = y1 if y1 >= 0 else 0
                    x2 = x2 if x2 <= img.shape[1] else img.shape[1]
                    y2 = y2 if y2 <= img.shape[0] else img.shape[0]
                    try:
                        listwk = img[y1:y2, x1:x2].reshape(-1).tolist()
                    except IndexError:
                        ## objectの中心が画面の外にはみ出してindex error になる場合があるのでその場合は省く
                        logger.warning("search range is out of images.")
                        continue
                    if len(listwk) == 0: continue
                    dfwk = pd.DataFrame(listwk, columns=["segmentation_color"])
                    dfwk["file_name"]     = files[i]
                    dfwk["category_name"] = dictwk["class"]

                    ## 背景色も判断するために画面の端もsampleする.
                    for _i in range(1,5):
                        dfwk["segmentation_color_end"+str(_i)] = img[0 if _i//2==0 else -1, 0 if _i%2==0 else -1]
                    list_df.append(dfwk.copy())

                count_file += 1
                if n_files is not None and count_file > n_files: break
                
            # 解析
            df = pd.concat(list_df, ignore_index=True, sort=False, axis=0)
            for x in ["segmentation_color"]+["segmentation_color_end"+str(_i) for _i in range(1,5)]: df[x] = df[x].astype(np.uint8)
            self.df_ndds = df.copy()
            ## 色のパターン
            dict_color = {x:None for x in df["category_name"].unique()}
            ## 背景色
            se = pd.concat([df["segmentation_color_end"+str(_i)] for _i in range(1,5)], ignore_index=True, sort=False, axis=0)
            se = se.value_counts().sort_values(ascending=False)
            dict_color["__bg"] = se.index[0]
            logger.info(f"back ground color: {se.index[0]}")
            ## object color
            se = df.groupby("segmentation_color")["category_name"].value_counts()
            se.name = "count"
            dfwk = se.reset_index().copy().sort_values(["segmentation_color", "count"], ascending=False)
            se = dfwk.groupby("segmentation_color").size().sort_values(ascending=True) # あるオブジェクトの周りにある色を集めて、category_nameの候補が少ない順にソートする
            for seg_color in se.index:
                dfwkwk = dfwk.loc[(dfwk["segmentation_color"] == seg_color), :]
                for i_loc, cat_name in enumerate(dfwkwk["category_name"].values):
                    if dict_color.get(cat_name) is None:
                        dict_color[cat_name] = seg_color
                        logger.info(f'{cat_name} color: {seg_color} count:{dfwkwk["count"].iloc[i_loc]}')
                        break
            logger.info(f"\r\n{dict_color}")
            ## それでも埋まらない色がある場合はエラー
            if (np.array(list(dict_color.values())) == None).sum() > 0:
                raise Exception(f"We can't fill segmentation color.")

            # category 化
            if instance_merge is not None:
                for x in instance_merge.keys():
                    self.instances[x] = []
                    for y in instance_merge[x]:
                        self.instances[x].append(dict_color[y])
            else:
                self.instances = {x: [dict_color[x]] for x in dict_color}
        
        elif self.convert_mode == "is":
            self.read_object_setting(correct_dirpath(dirpath) + self.setting_json_fname)
        
        else:
            logger.raise_error(f"convert_mode: {convert_mode} is not expected mode.")

        logger.debug("END")


    def read_object_setting(self, json_file_path: str, ignore_disexist_object: bool=True):
        """
        create dictionary {class_name: instance_color}
        """
        logger.debug("START")
        fjson = json.load(open(json_file_path))
        # まず単純にclassと色のdictionaryを作成する
        instances_org = {}
        for dictwk in fjson["exported_objects"]:
            instances_org[dictwk["class"]] = self.ndds_instanceid_to_color(dictwk["segmentation_instance_id"])

        self.instances = {}
        if self.instance_merge is not None:
            for x in self.instance_merge.keys():
                self.instances[x] = []
                for y in self.instance_merge[x]: # ここはlist
                    try:
                        self.instances[x].append(instances_org[y])
                    except KeyError:
                        if ignore_disexist_object:
                            pass
                        else:
                            raise Exception(f'we can not fond: {dictwk}')
        else:
            self.instances = {x:[instances_org[x]] for x in instances_org.keys()}
        logger.debug("END")


    def __read_ndds_output(self, json_file_path: str, visibility_threshold: float=0.1):
        logger.debug("START")

        #  json load
        fjson = json.load(open(json_file_path))

        # image file
        image_path = json_file_path.replace(".json", ".png")
        image_name = json_file_path[json_file_path.rfind("/")+1:].replace(".json", ".png")

        # mode によって読み込む画像を変更する
        img = None
        if   self.convert_mode == "cs":
            img = cv2.cvtColor(cv2.imread(json_file_path.replace(".json", ".cs.png")), cv2.COLOR_BGR2GRAY)
        elif self.convert_mode == "is":
            img = cv2.imread(json_file_path.replace(".json", ".is.png"))
            if tuple(np.unique(img)) == tuple(np.array([0], dtype=np.uint8)):
                ## is の判断の場合、画像が壊れている場合がある。壊れている画像はオール0の値の画像である
                logger.warning(f"is.png is broken. ignore image: {image_path}")
                return None
        height = img.shape[0]
        width  = img.shape[1]

        for name in self.instances.keys():
            # objects
            se = pd.Series(dtype=object)
            se["image_path"] = image_path
            se["image_name"] = image_name
            se["height"]     = height
            se["width"]      = width
            se["category_name"] = name
            # visivility threshold
            thre = 0.
            if self.instance_merge is not None:
                listwk = self.instance_merge[name]
                for dictwk in fjson["objects"]:
                    if dictwk["class"] in listwk: thre += dictwk["visibility"]
                thre = thre / len(listwk) # 複数のobject で1 classを表現している場合は、平均を取る
            else:
                for dictwk in fjson["object"]:
                    if dictwk["class"] == name:
                        thre = dictwk["visibility"]
                        break
            if thre < visibility_threshold: continue
            ## instance の処理
            if   self.convert_mode == "cs":
                ndf = np.isin(img, self.instances[name]).astype(np.uint8)
            elif self.convert_mode == "is":
                ndf = np.zeros_like(img[:, :, 0]).astype(np.int32)
                for ndfwk in self.instances[name]:
                    ndf = ndf + cv2.inRange(img, ndfwk, ndfwk)
                ndf = ndf.astype(np.uint8)
            ## 何も画像がない場合
            if np.sum(ndf) == 0: continue
            ## bounding box (自作) の処理
            x,y,w,h = None, None, None, None
            bbox_name = self.bboxes.get(name) if self.bboxes is not None else None
            if bbox_name is not None:
                for _object in fjson["objects"]:
                    if _object["class"] == bbox_name:
                        bbox = [_object["bounding_box"]["top_left"][::-1], _object["bounding_box"]["bottom_right"][::-1]] #x1,y1,x2,y2
                        for _i in range(2):
                            for _j in range(2):
                                if bbox[_i][_j] < 0: bbox[_i][_j] = 0
                            if bbox[_i][0] > width:  bbox[_i][0] = width
                            if bbox[_i][1] > height: bbox[_i][1] = height
                        x,y,w,h = bbox[0][0], bbox[0][1], bbox[1][0] - bbox[0][0], bbox[1][1] - bbox[0][1]
                        break
            else:
                x,y,w,h = cv2.boundingRect(ndf)
            if x is None: raise Exception("bbox is null.")
            se["bbox"] = (x,y,w,h,)
            ## segmmentation の処理
            ndf_seg = ndf.copy()
            if self.segmentations is not None and self.segmentations.get(name) is not None:
                seginfo = self.segmentations.get(name)
                for _object in fjson["objects"]:
                    if _object["class"] == seginfo["name"]:
                        if seginfo["type"] == "inbox":
                            bbox = [_object["bounding_box"]["top_left"][::-1], _object["bounding_box"]["bottom_right"][::-1]] #x1,y1,x2,y2
                            for _i in range(2):
                                for _j in range(2):
                                    if bbox[_i][_j] < 0: bbox[_i][_j] = 0
                                if bbox[_i][0] > width:  bbox[_i][0] = width
                                if bbox[_i][1] > height: bbox[_i][1] = height
                            p1, p2 = bbox[0], bbox[1]
                            ndf_seg_bool = np.zeros_like(ndf_seg).astype(bool)
                            ndf_seg_bool[int(p1[1]):int(p2[1]+1), int(p1[0]):int(p2[0]+1)] = True
                            ndf_seg[~ndf_seg_bool] = 0
                        else:
                            raise Exception(f'unexpected type: {seginfo["type"]}')
            contours = cv2.findContours(ndf_seg, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]
            se["segmentation"] = [contour.reshape(-1) for contour in contours]
            ## keypoint の処理
            se_key    = pd.Series(dtype=object)
            dict_keys = self.keypoints.get(name)
            for _object in fjson["objects"]:
                if dict_keys is not None and dict_keys.get(_object["class"]) is not None:
                    list_keys = dict_keys.get(_object["class"])
                    for info_key in list_keys:
                        p_key, vis = None, None
                        if   info_key["type"] == "center":
                            p1    = _object["bounding_box"]["top_left"][::-1] #y,xになっているので、x,yに入れ替える
                            p2    = _object["bounding_box"]["bottom_right"][::-1] #y,xになっているので、x,yに入れ替える
                            p_key = (int((p1[0]+p2[0])/2.), int((p1[1]+p2[1])/2.), )
                            vis   = 1 if ndf[p_key[1], p_key[0]] == 0 else 2
                            se_key[info_key["name"]] = (p_key[0], p_key[1], vis, ) # keypoint の x, y, visibility(0 or 1 or 2). 0:ラベルがない,1:ラベルがあって見えない,2:ラベルがあって見える
                        elif info_key["type"] in ["left", "tleft", "bleft"]:
                            p1    = _object["bounding_box"]["top_left"][::-1] #y,xになっているので、x,yに入れ替える
                            p2    = _object["bounding_box"]["bottom_right"][::-1] #y,xになっているので、x,yに入れ替える
                            _y    = -1
                            if   info_key["type"] ==  "left": _y = int((p1[1]+p2[1])/2.)
                            elif info_key["type"] == "tleft": _y = int(p1[1])
                            elif info_key["type"] == "bleft": _y = int(p2[1])
                            p_key = (int(p1[0]), _y, )
                            vis   = 1 if ndf[p_key[1], p_key[0]] == 0 else 2
                            se_key[info_key["name"]] = (p_key[0], p_key[1], vis, )
                        elif info_key["type"] in ["right", "tright", "bright"]:
                            p1    = _object["bounding_box"]["top_left"][::-1] #y,xになっているので、x,yに入れ替える
                            p2    = _object["bounding_box"]["bottom_right"][::-1] #y,xになっているので、x,yに入れ替える
                            _y    = -1
                            if   info_key["type"] ==  "right": _y = int((p1[1]+p2[1])/2.)
                            elif info_key["type"] == "tright": _y = int(p1[1])
                            elif info_key["type"] == "bright": _y = int(p2[1])
                            p_key = (int(p2[0]), _y, )
                            vis   = 1 if ndf[p_key[1], p_key[0]] == 0 else 2
                            se_key[info_key["name"]] = (p_key[0], p_key[1], vis, )
                        elif info_key["type"] == "left_seg":
                            try:
                                tate = np.where(ndf > 0)[0] # np.whereは縦, 横
                                yoko = np.where(ndf > 0)[1]
                                p_key = [yoko.min(), tate[yoko.argmin()]]
                                vis   = 2
                            except IndexError:
                                ## segmentation が 他のobject に隠れていた場合はBboxから計算する
                                p1    = _object["bounding_box"]["top_left"][::-1] #y,xになっているので、x,yに入れ替える
                                p2    = _object["bounding_box"]["bottom_right"][::-1] #y,xになっているので、x,yに入れ替える
                                p_key = (int(p1[0]), int((p1[1]+p2[1])/2.), )
                                vis   = 1
                            se_key[info_key["name"]] = (p_key[0], p_key[1], vis, ) 
                        elif info_key["type"] == "right_seg":
                            try:
                                tate = np.where(ndf > 0)[0] # np.whereは縦, 横
                                yoko = np.where(ndf > 0)[1]
                                p_key = [yoko.max(), tate[yoko.argmax()]]
                                vis   = 2
                            except IndexError:
                                ## segmentation が 他のobject に隠れていた場合はBboxから計算する
                                p1    = _object["bounding_box"]["top_left"][::-1] #y,xになっているので、x,yに入れ替える
                                p2    = _object["bounding_box"]["bottom_right"][::-1] #y,xになっているので、x,yに入れ替える
                                p_key = (int(p2[0]), int((p1[1]+p2[1])/2.), )
                                vis   = 1
                            se_key[info_key["name"]] = (p_key[0], p_key[1], vis, )
                        else:
                            raise Exception(f"info_key: {info_key}")
            dictwk = se_key.to_dict()
            se["num_keypoints"] = len(dictwk)
            se["keypoints"] = dictwk #np.array([(dictwk[_x] if dictwk.get(_x) is not None else (0,0,0,)) for _x in self.keypoints[name]]).reshape(-1).tolist() if se_key.shape[0] > 0 else []
            # 画像情報を追加
            self.images = self.images.append(se, ignore_index=True)

        logger.debug("END")


    def read_ndds_output_all(self, visibility_threshold: float=0.1, max_count: int=None):
        """
        Read NDDS ouptut files and segment objects with color parameters got at set_base_parameter()
        Params::
            dirpath: NDDS output directory path.
        
        Usage::
            >>> ndds_to_coco = Ndds2Coco()
            >>> ndds_to_coco.set_base_parameter(
                    "~/my_ndds_output_path", n_files=100, 
                    instance_merge={"human1":["body1","shoes1","clothes1", ...], "human2":["body2","shoes2","clothes2", ...], "dog1":["dog_body1", "dog_collars1"]}, 
                    visibility_threshold = 0.2
                    )
            >>> ndds_to_coco.read_ndds_output_all("~/my_ndds_output_path")
        """
        logger.debug("START")

        for i, x in enumerate(self.__get_json_files(self.indir)):
            logger.info(f"read file: {x}")
            if max_count is not None and max_count < i: break
            self.__read_ndds_output(x, visibility_threshold=visibility_threshold)
        
        # Keypoint の合体
        if self.keypoints_merge is not None:
            df  = self.images.copy()
            ndf_addr = df.loc[:, "keypoints"] # 参照形式で修正する
            for _, dfwk in self.images.groupby("image_name"):
                for x in self.keypoints_merge.keys():
                    listwk = self.keypoints_merge[x]
                    ndf = dfwk.loc[dfwk["category_name"].isin(listwk), "keypoints"].values
                    if ndf.shape[0] > 0:
                        dict_org: dict = ndf[0].copy()
                        for dictwk in ndf[1:]:
                            dict_org.update(dictwk)
                        index = dfwk.index[dfwk["category_name"] == x][0]
                        ndf_addr[index] = dict_org # dataframe を更新する
            df["num_keypoints"] = df["keypoints"].apply(lambda x: len(x))
            self.images = df.copy()

        # 型の変換
        for x in ["height", "width", "num_keypoints"]:
            self.images[x] = self.images[x].astype(np.int32)

        logger.debug("END")
    

    def draw_infomation(self, src) -> np.ndarray:
        # int の場合は その index の画像を見せる
        if   type(src) == int:
            df = self.images.iloc[src:src+1].copy() # df の状態で取得する
        # string の場合はその画像に含まれる全てのinstanceの情報を表示する
        elif type(src) == str:
            df = self.images[self.images["image_name"] == src].copy()

        # 画像の読み込み
        img = cv2.imread(df["image_path"].iloc[0])

        for i in np.arange(df.shape[0]):
            se = df.iloc[i]
            # bounding box の描画
            x,y,w,h = se["bbox"]
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            # segmentation の描画
            imgwk = np.zeros_like(img)
            for seg in se["segmentation"]:
                img   = cv2.polylines(img,[seg.reshape(-1,1,2)],True,(0,0,0))
                imgwk = cv2.fillConvexPoly(imgwk, points=seg.reshape(-1, 2), color=(255,0,0))
            img = cv2.addWeighted(img, 1, imgwk, 0.8, 0)
        
        return img
    

    def sample_ouptut(self, n_images: int = 100, dirpath = "./sample_ouptut", exist_ok: bool=False, remake: bool=False):
        """
        Output images converted to Coco format. 

        Params::
            n_images: number of sample output images.
            dirpath: output directory path.
        
        Usage::
            >>> ndds_to_coco = Ndds2Coco()
            >>> ndds_to_coco.set_base_parameter(
                    "~/my_ndds_output_path", n_files=100, 
                    instance_merge={"human1":["body1","shoes1","clothes1", ...], "human2":["body2","shoes2","clothes2", ...], "dog1":["dog_body1", "dog_collars1"]}, 
                    visibility_threshold = 0.2
                    )
            >>> ndds_to_coco.read_ndds_output_all("~/my_ndds_output_path")
            >>> ndds_to_coco.sample_ouptut(n_images: int = 100, dirpath = "./sample_ouptut")
        """
        logger.info("START")
        makedirs(dirpath, exist_ok=exist_ok, remake=remake)
        ndf = self.images["image_name"].unique()
        for x in ndf[ np.random.permutation(np.arange(ndf.shape[0]))[:n_images] ]:
            img = self.draw_infomation(x)
            cv2.imwrite(dirpath + "/" + x, img)

        logger.info("END")


    def to_coco_format(self, category_merge: dict) -> str:
        logger.debug("START")

        json_dict = {}
        # info
        json_dict["info"] = self.info
        # licenses
        json_dict["licenses"] = []
        json_dict["licenses"].append({"url":"http://test","id":0,"name":"test license"})
        # images
        json_dict["images"] = []
        json_dict["annotations"] = []
        dict_category_id = {}
        dict_category_id_name = {}
        for i, x in enumerate(category_merge["categories"].keys()):
            dict_category_id_name[x] = i
            for y in category_merge["categories"][x]:
                if dict_category_id.get(y) is None: dict_category_id[y] = []
                dict_category_id[y].append(i) # {"corn1": [0], "corn2": [0], ..., coneX: [Y], ...}
        for i, (image_name, df) in enumerate(self.images.groupby("image_name")):
            dictwk = {}
            dictwk["license"]       = 0
            dictwk["file_name"]     = image_name
            dictwk["coco_url"]      = df["image_path"].iloc[0]
            dictwk["height"]        = int(df["height"].iloc[0]) # np.int32 形式だとjson変換でエラーになるため
            dictwk["width"]         = int(df["width" ].iloc[0]) # np.int32 形式だとjson変換でエラーになるため
            dictwk["date_captured"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            dictwk["flickr_url"]    = "http://test"
            dictwk["id"]            = i
            json_dict["images"].append(dictwk)

            for i_index in np.arange(df.shape[0]):
                se = df.iloc[i_index]
                if dict_category_id.get(se["category_name"]) is None: continue # coco json に必要ないcategory は飛ばす
                for category_id in dict_category_id[se["category_name"]]:
                    # instance 別の処理
                    dictwk = {}
                    dictwk["segmentation"]  = [_x.tolist() for _x in se["segmentation"]]
                    dictwk["area"]          = sum([cv2.contourArea(_x.reshape(-1, 1, 2)) for _x in se["segmentation"]])
                    dictwk["bbox"]          = list(se["bbox"])
                    dictwk["iscrowd"]       = 0
                    dictwk["image_id"]      = i
                    dictwk["category_id"]   = int(category_id) # np.int32 形式だとjson変換でエラーになるため
                    dictwk["id"]            = int(df.index[i_index]) # とりあえずself.images の index が一意なのでそれをつける
                    dictwk["num_keypoints"] = int(se["num_keypoints"])
                    dict_keys               = se["keypoints"]
                    dictwk["keypoints"]     = np.array([(dict_keys[_x] if dict_keys.get(_x) is not None else (0,0,0,)) for _x in category_merge["keypoints"]]).reshape(-1).tolist()
                    dictwk["keypoints"]     = [int(_x) for _x in dictwk["keypoints"]]
                    json_dict["annotations"].append(dictwk)
        
        # categories
        json_dict["categories"] = []
        for x in category_merge["supercategory"].keys():
            for y in category_merge["supercategory"][x]:
                json_dict["categories"].append({
                    "supercategory":x, "name":y, "id": dict_category_id_name[y], 
                    "keypoints":category_merge["keypoints"] if category_merge.get("keypoints") is not None else [],
                    "skeleton": category_merge["skelton"]   if category_merge.get("skelton") is not None else [],
                })

        logger.debug("END")
        return json.dumps(json_dict)
    

    def output_coco_format(self, out_json_path: str, category_merge: dict, out_images_dir: str=None):
        """
        Output coco format json file and images( only copy images from original files ).

        Params::
            dirpath: output directory path.
                dirpath / json / instances.json
                dirpath / images / *.png or *.jpg

            category_merge:
                dictionary variable for merge different instances into a category
                and merge differnt categories into a supercategory.
                    ex.
                        if set use like set_base_parameter(..., instance_merge={"human1":["body1","shoes1","clothes1", ...], "human2":["body2","shoes2","clothes2", ...], "dog1":["dog_body1", "dog_collars1"]}, ...), 
                        category_merge = {"supercategory":{"animal":["human", "dog", ...], "others":[...]}, 
                                          "categories"   :{"human" :["human1", "human2", ...], "dog":["dog1", ...] }}

        Usage::
            >>> ndds_to_coco = Ndds2Coco()
            >>> ndds_to_coco.set_base_parameter(
                    "~/my_ndds_output_path", n_files=100, 
                    instance_merge={"human1":["body1","shoes1","clothes1", ...], "human2":["body2","shoes2","clothes2", ...], "dog1":["dog_body1", "dog_collars1"]}, 
                    visibility_threshold = 0.2
                    )
            >>> ndds_to_coco.read_ndds_output_all("~/my_ndds_output_path")
            >>> ndds_to_coco.output_coco_format(
                    "./output_ndds_to_coco", 
                    {"supercategory":{"animal":["human", "dog", ...], "others":[...]}, 
                     "categories"   :{"human" :["human1", "human2", ...], "dog":["dog1", ...] }}
                )

        """
        logger.debug("START")

        # json output
        with open(out_json_path, mode="w") as f:
            f.write(self.to_coco_format(category_merge=category_merge))
        
        # image copy
        if out_images_dir is not None:
            out_images_dir = correct_dirpath(out_images_dir)
            os.makedirs(out_images_dir, exist_ok=True)
            for x in self.images["image_path"].unique():
                shutil.copy2(x, out_images_dir)

        logger.debug("END")



class CocoManager:
    """Coco format manager with pandas DataFrame"""

    def __init__(self):
        self.df_json   = pd.DataFrame()
        self.json      = {} # 直近の１つを保存する
        self.coco_info = {}


    @classmethod
    def json_to_df(cls, src):
        check_type(src, [str, dict])
        # json の load
        json_coco = {}
        if   type(src) == str:  json_coco = json.load(open(src))
        elif type(src) == dict: json_coco = src
        # json の構造定義
        df = pd.DataFrame(json_coco["images"])
        df.columns = ["images_"+x for x in df.columns]
        dfwk = pd.DataFrame(json_coco["annotations"])
        dfwk.columns = ["annotations_"+x for x in dfwk.columns]
        df = pd.merge(df, dfwk, how="left", left_on="images_id", right_on="annotations_image_id")
        dfwk = pd.DataFrame(json_coco["licenses"])
        dfwk.columns = ["licenses_"+x for x in dfwk.columns]
        df = pd.merge(df, dfwk, how="left", left_on="images_license", right_on="licenses_id")
        dfwk = pd.DataFrame(json_coco["categories"])
        dfwk.columns = ["categories_"+x for x in dfwk.columns]
        df = pd.merge(df, dfwk, how="left", left_on="annotations_category_id", right_on="categories_id")
        return df


    def check_index(self):
        # file name が同じで path が違うものがあればチェックする
        se = self.df_json.groupby("images_file_name")["images_coco_url"].apply(lambda x: x.unique())
        if (se.apply(lambda x: len(x) > 1)).sum() > 0:
            logger.warning(f"same file name: [{se[se.apply(lambda x: len(x) > 1)].index.values}]")


    def re_index(
        self, check_file_exist_dir: str=None, 
        keypoints: List[str]=None, skeleton: List[List[str]]=None
    ):
        # ファイルが存在しなければcoco から外す
        if check_file_exist_dir is not None:
            filelist = [os.path.basename(x) for x in get_file_list(check_file_exist_dir)]
            self.df_json = self.df_json[self.df_json["images_file_name"].isin(filelist)]

        # coco format に足りないカラムがあれば追加する
        for name, default_value in zip(
            ["categories_keypoints", "categories_skeleton", "licenses_name", "licenses_url"], 
            [[], [], np.nan, np.nan]
        ):
            if (self.df_json.columns == name).sum() == 0:
                self.df_json[name] = [default_value for _ in np.arange(self.df_json.shape[0])]

        # image id
        dictwk = {x:i  for i, x in enumerate(np.sort(self.df_json["images_coco_url"].unique()))}
        self.df_json["images_id"]            = self.df_json["images_coco_url"].map(dictwk)
        self.df_json["annotations_image_id"] = self.df_json["images_coco_url"].map(dictwk)
        # license id
        self.df_json["licenses_name"] = self.df_json["licenses_name"].fillna("test license")
        self.df_json["licenses_url"]  = self.df_json["licenses_url"]. fillna("http://test")
        dictwk = {x:i  for i, x in enumerate(np.sort(self.df_json["licenses_name"].unique()))}
        self.df_json["images_license"] = self.df_json["licenses_name"].map(dictwk)
        self.df_json["licenses_id"]    = self.df_json["licenses_name"].map(dictwk)
        # category id ( super category と category name で一意とする )
        if self.df_json.columns.isin(["categories_supercategory"]).sum() == 0:
            self.df_json["categories_supercategory"] = self.df_json["categories_name"].copy()
        self.df_json["__work"] = (self.df_json["categories_supercategory"].astype(str) + "_" + self.df_json["categories_name"].astype(str)).copy()
        dictwk = {x:i  for i, x in enumerate(np.sort(self.df_json["__work"].unique()))}
        self.df_json["annotations_category_id"] = self.df_json["__work"].map(dictwk)
        self.df_json["categories_id"]           = self.df_json["__work"].map(dictwk)
        self.df_json = self.df_json.drop(columns=["__work"])
        # annotations id ( df の index に相当する)
        self.df_json = self.df_json.reset_index(drop=True)
        self.df_json["annotations_id"] = self.df_json.index.values
        # Keypoint の統合
        if keypoints is not None:
            self.df_json["annotations_keypoints"] = self.df_json.apply(
                lambda x: np.array(
                    [x["annotations_keypoints"][np.where(np.array(x["categories_keypoints"]) == y)[0].min()*3:np.where(np.array(x["categories_keypoints"]) == y)[0].min()*3+3]
                    if y in x["categories_keypoints"] else [0,0,0] for y in keypoints]
                ).reshape(-1).tolist(), axis=1
            )
            self.df_json["categories_keypoints"] = self.df_json["categories_keypoints"].apply(lambda x: keypoints)
            skeleton = np.array(skeleton).copy()
            for i, x in enumerate(keypoints):
                skeleton[skeleton == x] = i
            skeleton = skeleton.astype(int)
            self.df_json["categories_skeleton"] = self.df_json["categories_skeleton"].apply(lambda x: skeleton.tolist())


    def organize_segmentation(self):
        """
        Segmentation の annotation を使って augmentation をする場合、
        変な annotation になっているとエラーが発生するため、整理するための関数
        """
        ndf_json  = self.df_json.values.copy()
        index_seg = np.where(self.df_json.columns == "annotations_segmentation")[0].min()
        index_w   = np.where(self.df_json.columns == "images_width")[0].min()
        index_h   = np.where(self.df_json.columns == "images_height")[0].min()
        for i, (h, w, list_seg,) in enumerate(ndf_json[:, [index_h, index_w, index_seg]]):
            ndf = np.zeros((int(h), int(w), 3)).astype(np.uint8)
            re_seg = []
            for seg in list_seg:
                # segmentation を 線を繋いで描く
                ndf = cv2.polylines(ndf, [np.array(seg).reshape(-1,1,2).astype(np.int32)], True, (255,255,255))
                # 線を描いた後、一番外の輪郭を取得する
                contours, _ = cv2.findContours(ndf[:, :, 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                re_seg.append(contours[0].reshape(-1).astype(int).tolist())
            ndf_json[i, index_seg] = re_seg


    def add_json(self, src, root_image: str=None):
        check_type(src, [str, dict])
        json_coco = {}
        if   type(src) == str:  json_coco = json.load(open(src))
        elif type(src) == dict: json_coco = src
        self.json = json_coco.copy()
        if json_coco.get("licenses") is None or len(json_coco["licenses"]) == 0:
            json_coco["licenses"] = [{'url': 'http://test', 'id': 0, 'name': 'test license'}]
        try:
            self.coco_info = json_coco["info"]
        except KeyError:
            print("'src' file or dictionary is not found 'info' key. so, skip this src.")
            return None
        df = self.json_to_df(json_coco)
        if root_image is not None: df["images_coco_url"] = correct_dirpath(root_image) + df["images_file_name"]
        self.df_json = pd.concat([self.df_json, df], axis=0, ignore_index=True, sort=False)
        self.check_index()
        self.re_index()
    

    def check_file_exist(self, img_dir: str=None):
        list_target = []
        if img_dir is None:
            for x in self.df_json["images_coco_url"].unique():
                if os.path.exists(x):
                    list_target.append(x)
            self.df_json = self.df_json.loc[self.df_json["images_coco_url"].isin(list_target), :]
        else:
            img_dir = correct_dirpath(img_dir)
            for x in self.df_json["images_file_name"].unique():
                if os.path.exists(img_dir + x):
                    list_target.append(x)
            self.df_json = self.df_json.loc[self.df_json["images_file_name"].isin(list_target), :]
    

    def rmkeypoints(self, list_key_names: List[str], list_key_skeleton:List[List[str]]):
        """
        Params::
            list_key_names: 残すKeypointの名前のList
        """
        df = self.df_json.copy()
        df["annotations_keypoints"] = df[["annotations_keypoints","categories_keypoints"]].apply(
            lambda x: np.array(x["annotations_keypoints"]).reshape(-1, 3)[np.isin(np.array(x["categories_keypoints"]), list_key_names)].reshape(-1).tolist(), 
            axis=1
        ).copy()
        df["annotations_num_keypoints"] = df["annotations_keypoints"].apply(lambda x: (np.array(x).reshape(-1, 3)[::, 2] > 0).sum())
        df["categories_keypoints"] = df["categories_keypoints"].apply(lambda x: np.array(x)[np.isin(np.array(x), list_key_names)].tolist()).copy()
        df["categories_skeleton"]  = df["categories_keypoints"].apply(lambda x: [np.where(np.isin(np.array(x), _x))[0].tolist() for _x in list_key_skeleton])
        self.df_json = df.copy()


    def concat_segmentation(self, category_name: str):
        """
        category_name に指定したクラスの segmentation を images_coco_url 単位で結合する
        """
        df = self.df_json.copy()
        df = df[df["categories_name"] == category_name]
        df_rep = pd.DataFrame()
        for _, dfwk in df.groupby("images_coco_url"):
            se = dfwk.iloc[0].copy()
            list_seg = []
            for segs in dfwk["annotations_segmentation"].values:
                for seg in segs:
                    list_seg.append([int(x) for x in seg])
            se["annotations_segmentation"] = list_seg
            bbox = np.concatenate(list_seg).reshape(-1, 2)
            se["annotations_bbox"] = [bbox[:, 0].min(), bbox[:, 1].min(), bbox[:, 0].max() - bbox[:, 0].min(), bbox[:, 1].max() - bbox[:, 1].min()]
            se["annotations_bbox"] = [int(x) for x in se["annotations_bbox"]]
            se["annotations_area"] = int(dfwk["annotations_area"].sum())
            df_rep = df_rep.append(se, ignore_index=True, sort=False)
        df = self.df_json.copy()
        df = df[df["categories_name"] != category_name]
        df = pd.concat([df, df_rep], axis=0, ignore_index=True, sort=False)
        self.df_json = df.copy()
        self.re_index()
    

    def copy_annotation(self, catname_copy_from: str, catname_copy_to: str, colnames: List[str]):
        """
        images_coco_url 単位である category name の○○を別の category name の○○に copyする
        """
        df = self.df_json.copy()
        ## ある image に instance が複数ある場合は最初の annotation を取ってくる
        df_from = df[df["categories_name"] == catname_copy_from].groupby("images_coco_url").first()
        df_from.columns = [x + "_from" for x in df_from.columns]
        df_from = df_from.reset_index()
        df_to   = df[df["categories_name"] == catname_copy_to  ].copy()
        for colname in colnames:
            df_to = pd.merge(df_to, df_from[["images_coco_url",colname+"_from"]].copy(), how="left", on="images_coco_url")
            df_to[colname] = df_to[colname + "_from"].copy()
            if df_to[colname].isna().sum() > 0: raise Exception(f'{colname} has nan')
            df_to = df_to.drop(columns=[colname + "_from"])
        df = df[df["categories_name"] != catname_copy_to]
        df = pd.concat([df, df_to], axis=0, ignore_index=True, sort=False)
        self.df_json = df.copy()
        self.re_index()



    def to_coco_format(self, df_json: pd.DataFrame = None):
        df_json = df_json.copy() if df_json is not None else self.df_json.copy()
        json_dict = {}
        json_dict["info"] = self.coco_info
        for _name in ["images", "annotations", "licenses", "categories"]:
            df = df_json.loc[:, df_json.columns.str.contains("^"+_name+"_", regex=True)].copy()
            df.columns = df.columns.str[len(_name+"_"):]
            df = df.fillna("%%null%%")
            json_dict[_name] = df.groupby("id").first().reset_index().apply(lambda x: x.to_dict(), axis=1).to_list()
        strjson = json.dumps(json_dict)
        return strjson.replace('"%%null%%"', 'null')


    def save(self, filepath: str, save_images_path: str = None, exist_ok = False, remake = False):
        if save_images_path is not None:
            save_images_path = correct_dirpath(save_images_path)
            makedirs(save_images_path, exist_ok = exist_ok, remake = remake)
            # 同名ファイルはrename する
            dfwk = self.df_json.groupby(["images_file_name","images_coco_url"]).size().reset_index()
            dfwk["images_file_name"] = dfwk.groupby("images_file_name")["images_file_name"].apply(lambda x: pd.Series([get_filename(y)+"."+str(i)+"."+y.split(".")[-1] for i, y in enumerate(x)]) if x.shape[0] > 1 else x).values
            self.df_json["images_file_name"] = self.df_json["images_coco_url"].map(dfwk.set_index("images_coco_url")["images_file_name"].to_dict())
            # image copy
            for x, y in dfwk[["images_coco_url", "images_file_name"]].values:
                shutil.copy2(x, save_images_path+y)
            # coco url の file name を変更しておく
            self.df_json["images_coco_url"] = save_images_path + self.df_json["images_file_name"]
            self.re_index()

        with open(filepath, "w") as f:
            f.write(self.to_coco_format())
    

    def draw_infomation(self, src, imgpath: str = None, is_show: bool=True, is_anno_name: bool=False) -> np.ndarray:
        # int の場合は その index の画像を見せる
        if   type(src) == int:
            df = self.df_json.iloc[src:src+1].copy() # df の状態で取得する
        # string の場合はその画像に含まれる全てのinstanceの情報を表示する
        elif type(src) == str:
            df = self.df_json[self.df_json["images_file_name"] == src].copy()

        # 画像の読み込み
        url = df["images_coco_url"].iloc[0] if imgpath is None else correct_dirpath(imgpath) + df["images_file_name"].iloc[0]
        img = cv2.imread(url)

        for i in np.arange(df.shape[0]):
            se = df.iloc[i]
            # bounding box の描画
            x,y,w,h = se["annotations_bbox"]
            img = cv2.rectangle(img,(int(x),int(y)),(int(x+w),int(y+h)),(0,255,0),2)
            if is_anno_name:
                cv2.putText(img, se['categories_name'], (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), thickness=2)
            # segmentation の描画
            imgwk = np.zeros_like(img)
            for listwk in se["annotations_segmentation"]:
                seg = np.array(listwk)
                img   = cv2.polylines(img,[seg.reshape(-1,1,2).astype(np.int32)],True,(0,0,0))
                imgwk = cv2.fillConvexPoly(imgwk, points=seg.reshape(-1, 2).astype(np.int32), color=(255,0,0))
            img = cv2.addWeighted(img, 1, imgwk, 0.8, 0)
            # keypoint の描画
            ann_key_pnt = np.array(se["annotations_keypoints"]).reshape(-1, 3).astype(int)
            for j, (x, y, v, ) in enumerate(ann_key_pnt):
                color = (0, 0, 255) if v == 2 else ((255, 0, 0) if v == 1 else (0, 0, 0,))
                if v > 0:
                    img = cv2.circle(img, (int(x), int(y)), 5, color, thickness=-1)
                    if is_anno_name:
                        cv2.putText(img, se['categories_keypoints'][j], (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=1)
            for p1, p2 in se["categories_skeleton"]:
                img = cv2.line(img, tuple(ann_key_pnt[p1][:2]), tuple(ann_key_pnt[p2][:2]), (0, 0, 255))

        if is_show:
            cv2.imshow("sample", img)
            cv2.waitKey(0)

        return img
    

    def output_draw_infomation(self, outdir: str, imgpath: str = None, exist_ok: bool=False, remake: bool=False, is_anno_name: bool=False):
        outdir = correct_dirpath(outdir)
        makedirs(outdir, exist_ok=exist_ok, remake=remake)
        for x in self.df_json["images_file_name"].unique():
            logger.info(f"write coco annotation in image: {x}")
            img = self.draw_infomation(x, imgpath=imgpath, is_show=False, is_anno_name=is_anno_name)
            cv2.imwrite(outdir + x, img)


    def scale_bbox(self, target: dict = {}, padding_all: int=None):
        """
        bbox を広げたり縮めたりする
        Params:
            target:
                dict: {categories_name: scale} の形式で指定する. 複数OK, 
                sclae: int or float. int の場合は固定pixel, scale は bboxを等倍する
            padding_all:
                None でないなら target を無視しして全ての annotation に適用する
        """
        logger.info("START")
        df = self.df_json.copy()
        if padding_all is not None and type(padding_all) in [float, int]:
            scale = padding_all
            if   type(scale) == int:
                df["annotations_bbox"] = df["annotations_bbox"].apply(lambda x: [x[0] - scale, x[1] - scale, x[2] + 2*scale, x[3] + 2*scale])
            elif type(scale) == float:
                df["annotations_bbox"] = df["annotations_bbox"].apply(
                    lambda x: [x[0] - (x[2] * scale - x[2])/2., x[1] - (x[3] * scale - x[3])/2., x[2] * scale, x[3] * scale]
                )
        else:
            for x in target.keys():
                dfwk = df[df["categories_name"] == x].copy()
                scale = target[x]
                if   type(scale) == int:
                    dfwk["annotations_bbox"] = dfwk["annotations_bbox"].apply(lambda x: [x[0] - scale, x[1] - scale, x[2] + 2*scale, x[3] + 2*scale])
                elif type(scale) == float:
                    dfwk["annotations_bbox"] = dfwk["annotations_bbox"].apply(
                        lambda x: [x[0] - (x[2] * scale - x[2])/2., x[1] - (x[3] * scale - x[3])/2., x[2] * scale, x[3] * scale]
                    )
                df.loc[dfwk.index, "annotations_bbox"] = dfwk["annotations_bbox"].copy()
        # scale の結果、bbox が画面からはみ出している場合があるので修正する
        df = self.fix_bbox_value(df)
        self.df_json = df.copy()
        logger.info("END")


    @classmethod
    def fix_bbox_value(cls, df: pd.DataFrame) -> pd.DataFrame:
        ## 参照形式で修正する.
        df = df.copy()
        for listwk, w, h in df[["annotations_bbox", "images_width", "images_height"]].values:
            ### 先に w,h を計算しないと値がおかしくなる
            if listwk[0] < 0:             listwk[2] = int(listwk[2] + listwk[0]) # 始点xが左側の枠外の場合
            if listwk[0] + listwk[2] > w: listwk[2] = int(w  - listwk[0]) # 始点xが枠内で始点x+幅が右側の枠外になる場合.
            if listwk[2] > w:             listwk[2] = int(w)              # それでもまだ大きい場
            if listwk[1] < 0:             listwk[3] = int(listwk[3] + listwk[1]) # 始点yが下側の枠外の場合
            if listwk[1] + listwk[3] > h: listwk[3] = int(h  - listwk[1]) # 始点yが枠内で始点y+高さが上側の枠外になる場合.
            if listwk[3] > h:             listwk[3] = int(h)              # それでもまだ大きい場
            listwk[0] = 0 if listwk[0] < 0 else listwk[0] # 0 以下は0に
            listwk[1] = 0 if listwk[1] < 0 else listwk[1]
        df["annotations_bbox"] = df["annotations_bbox"].apply(lambda x: [int(_x) for _x in x])
        return df


    def padding_image_and_re_annotation(
        self, add_padding: int, root_image: str, outdir: str, 
        outfilename: str="output.json", fill_color=(0,0,0,), exist_ok: bool=False, remake: bool=False
    ):
        """
        Params::
            fill_color: (0,0,0,), など色を指定するか "random" にする
        """
        logger.info("START")
        # annotation をずらず
        df = self.df_json.copy()
        df["images_height"] = df["images_height"] + 2 * add_padding
        df["images_width"]  = df["images_width"]  + 2 * add_padding
        df["annotations_bbox"] = df["annotations_bbox"].map(lambda x: [_x+(add_padding if i in [0,1] else 0) for i, _x in enumerate(x)]) # x,yにだけadd_paddingを足す
        df["annotations_segmentation"] = df["annotations_segmentation"].map(lambda x: [[__x+add_padding for __x in _x] for _x in x])

        # 画像を切り抜く
        root_image = correct_dirpath(root_image)
        outdir     = correct_dirpath(outdir)
        makedirs(outdir, exist_ok=exist_ok, remake=remake)
        for imgname, dfwk in df.groupby(["images_file_name"]):
            logger.info(f"resize image name: {imgname}")
            img = cv2.imread(root_image + imgname)
            img = cv2.copyMakeBorder(
                img, add_padding, add_padding, add_padding, add_padding,
                cv2.BORDER_CONSTANT, value=(np.random.randint(0, 255, 3).tolist() if type(fill_color) == str and fill_color == "random" else fill_color),
            )
            filepath = outdir + imgname + ".png"
            # 画像を保存する
            cv2.imwrite(filepath, img)
            # coco の 情報 を書き換える
            df.loc[dfwk.index, "images_file_name"] = os.path.basename(filepath)
            df.loc[dfwk.index, "images_coco_url" ] = filepath
        df["images_date_captured"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.df_json = df.copy()
        self.re_index()
        with open(outdir + outfilename, "w") as f:
            f.write(self.to_coco_format())
        logger.info("END")


    def crop_image_and_re_annotation(
        self, crop_by: dict, root_image: str, outfilename: str, 
        outdir: str="./output_crop_images"
    ):
        """
        crop_by で指定された方法で画像を切り抜いてre_annotationする
        Params::
            crop_by:
                bounding box: {"bbox":["color_cone"]} のように指定
        """
        logger.info("START")
        list_df_ret = []
        for method in crop_by.keys():
            df = self.df_json.copy()
            if method == "bbox":
                for ann in crop_by[method]:
                    # annotation 単位でループする
                    for imgid, (x,y,w,h,) in df[df["categories_name"] == ann][["images_id", "annotations_bbox"]].copy().values:
                        logger.info(f"image id: {imgid}")
                        df_ret = df[(df["images_id"] == imgid)].copy()
                        str_resize = "_".join([str(_y) for _y in [int(x), int(y), int(x+w), int(y+h)]])
                        df_ret["__resize"] = str_resize # 最後に画像を切り抜くために箱を用意する
                        bboxwk = [int(x), int(y), 0, 0]
                        ## height, width
                        df_ret["images_height"] = int(h)
                        df_ret["images_width"]  = int(w)
                        ## bbox の 修正(細かな補正は for文の外で行う)
                        df_ret["annotations_bbox"] = df_ret["annotations_bbox"].apply(lambda _x: [_y - bboxwk[_i] for _i, _y in enumerate(_x)])
                        ## segmentation の 修正
                        ### segmentation : [[x1, y1, x2, y2, ...], [x1', y1', x2', y2', ...], ]
                        df_ret["annotations_segmentation"] = df_ret["annotations_segmentation"].apply(lambda _x: [[_y - bboxwk[_i%2] for _i, _y in enumerate(_listwk)] for _listwk in _x])
                        ndf = df_ret["annotations_segmentation"].values # ndarray に渡して参照形式で修正する. ※汚いけど...
                        for _i in np.arange(ndf.shape[0]):
                            _list = ndf[_i] # ここでlistのlist[[854, 121, 855, 120, 856, 120, 857, 120, ...], [...]]
                            for _listwk in _list:
                                for _j in np.arange(len(_listwk)):
                                    ### 偶数はx座標, 奇数はy座標
                                    if _j % 2 == 0:
                                        _listwk[_j] = 0      if _listwk[_j] < 0      else _listwk[_j]
                                        _listwk[_j] = int(w) if _listwk[_j] > int(w) else _listwk[_j]
                                    else:
                                        _listwk[_j] = 0      if _listwk[_j] < 0      else _listwk[_j]
                                        _listwk[_j] = int(h) if _listwk[_j] > int(h) else _listwk[_j]
                        ## Keypoint の修正
                        if len(df_ret["categories_keypoints"].iloc[0]) > 0:
                            ndf = df_ret["annotations_keypoints"].values
                            ndf = np.concatenate([[x] for x in ndf], axis=0)
                            ndf = ndf.reshape(ndf.shape[0], -1, 3)
                            ndf[:, :, 0] = ndf[:, :, 0] - bboxwk[0] # 切り取り位置を引く
                            ndf[:, :, 1] = ndf[:, :, 1] - bboxwk[1] # 切り取り位置を引く
                            ndf[:, :, 2][ndf[:, :, 0] <= 0] = 0 # はみ出したKeypointはvisを0にする
                            ndf[:, :, 2][ndf[:, :, 1] <= 0] = 0 # はみ出したKeypointはvisを0にする
                            ndf[:, :, 0][ndf[:, :, 2] == 0] = 0 # vis=0のkeypointはx, y を 0 にする
                            ndf[:, :, 1][ndf[:, :, 2] == 0] = 0 # vis=0のkeypointはx, y を 0 にする
                            ndf_nkpts = (ndf[:, :, 2] > 0).sum(axis=1).reshape(-1)
                            ndf = ndf.reshape(ndf.shape[0], -1)
                            sewk = pd.Series(dtype=object)
                            for i, index in enumerate(df_ret.index): sewk[str(index)] = ndf[i].tolist()
                            sewk.index = df_ret.index.copy()
                            df_ret["annotations_keypoints"]     = sewk
                            df_ret["annotations_num_keypoints"] = ndf_nkpts.tolist()
                        list_df_ret.append(df_ret.copy())
        df_ret = pd.concat(list_df_ret, axis=0, ignore_index=True, sort=False)
        # bbox の枠外などの補正
        ## 始点が枠外は除外
        boolwk = (df_ret["annotations_bbox"].map(lambda x: x[0]) >= df_ret["images_width"]) | (df_ret["annotations_bbox"].map(lambda x: x[1]) >= df_ret["images_height"])
        df_ret = df_ret.loc[~boolwk, :]
        ## 終点が枠外は除外
        boolwk = (df_ret["annotations_bbox"].map(lambda x: x[0]+x[2]) <= 0) | (df_ret["annotations_bbox"].map(lambda x: x[1]+x[3]) <= 0)
        df_ret = df_ret.loc[~boolwk, :]
        ## ndarray に渡して参照形式で修正する. ※汚いけど...
        ndf, ndf_w, ndf_h = df_ret["annotations_bbox"].values, df_ret["images_width"].values, df_ret["images_height"].values 
        for i in np.arange(ndf.shape[0]):
            listwk = ndf[i]
            ### 先に w,h を計算しないと値がおかしくなる
            if listwk[0] < 0:                    listwk[2] = int(listwk[2] + listwk[0]) # 始点xが左側の枠外の場合
            if listwk[0] + listwk[2] > ndf_w[i]: listwk[2] = int(ndf_w[i]  - listwk[0]) # 始点xが枠内で始点x+幅が右側の枠外になる場合.
            if listwk[2] > ndf_w[i]:             listwk[2] = int(ndf_w[i])              # それでもまだ大きい場
            if listwk[1] < 0:                    listwk[3] = int(listwk[3] + listwk[1]) # 始点yが下側の枠外の場合
            if listwk[1] + listwk[3] > ndf_h[i]: listwk[3] = int(ndf_h[i]  - listwk[1]) # 始点yが枠内で始点y+高さが上側の枠外になる場合.
            if listwk[3] > ndf_h[i]:             listwk[3] = int(ndf_h[i])              # それでもまだ大きい場
            listwk[0] = 0 if listwk[0] < 0 else listwk[0] # 0 以下は0に
            listwk[1] = 0 if listwk[1] < 0 else listwk[1]

        # 画像を切り抜く. 新しい画像を作成し、名前を変える
        root_image = correct_dirpath(root_image)
        outdir     = correct_dirpath(outdir)
        makedirs(outdir, exist_ok=True, remake=True)
        for (imgname, str_resize,), dfwk in df_ret.groupby(["images_file_name", "__resize"]):
            logger.info(f"resize image name: {imgname}")
            x1, y1, x2, y2 = [int(x) for x in str_resize.split("_")]
            img = cv2.imread(root_image + imgname)
            img = img[y1:y2, x1:x2, :]
            filepath = outdir + imgname + "." + str_resize + ".png"
            # 画像を保存する
            cv2.imwrite(filepath, img)
            # coco の 情報 を書き換える
            df_ret.loc[dfwk.index, "images_file_name"] = os.path.basename(filepath)
            df_ret.loc[dfwk.index, "images_coco_url" ] = filepath
        df_ret["images_date_captured"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df_ret = df_ret.drop(columns=["__resize"])

        # 画面外にはみ出したannotationを修正する
        ## 細かい修正は↑で行っているので、ここでは bbox = (0, y, 0, h) or (x, 0, w, 0) になっている行を省く
        boolwk = np.array([False] * df_ret.shape[0])
        boolwk = boolwk | (( df_ret["annotations_bbox"].map(lambda x: x[0]) == 0 ) & ( df_ret["annotations_bbox"].map(lambda x: x[2]) == 0 ))
        boolwk = boolwk | (( df_ret["annotations_bbox"].map(lambda x: x[1]) == 0 ) & ( df_ret["annotations_bbox"].map(lambda x: x[3]) == 0 ))
        df_ret = df_ret.loc[~boolwk, :] 

        self.df_json = df_ret.copy()
        self.re_index()

        with open(outdir + outfilename, "w") as f:
            f.write(self.to_coco_format())
        
        logger.info("END")


    def split_validation_data(self, path_json_train: str, path_json_valid: str, size: float=0.1):
        df = self.df_json.copy()
        ndf_fname = df["images_file_name"].unique()
        ndf_fname = np.random.permutation(ndf_fname)
        size = int(ndf_fname.shape[0] * size)
        df_train = df[df["images_file_name"].isin(ndf_fname[:-size ])].copy()
        df_valid = df[df["images_file_name"].isin(ndf_fname[ -size:])].copy()
        # train
        self.df_json = df_train
        self.re_index()
        self.save(path_json_train)
        # train
        self.df_json = df_valid
        self.re_index()
        self.save(path_json_valid)
        # 元に戻す
        self.df_json = df



class Labelme2Coco:

    def __init__(
        self, dirpath_json: str, dirpath_img: str, 
        categories_name: List[str], keypoints: List[str] = None, 
        keypoints_belong: dict=None, skelton: List[List[str]] = None
    ):
        self.dirpath_json = correct_dirpath(dirpath_json)
        self.dirpath_img  = correct_dirpath(dirpath_img)
        self.categories_name      = categories_name
        self.index_categories_name = {x:i for i, x in enumerate(categories_name)}
        self.keypoints = keypoints
        self.keypoints_belong = keypoints_belong
        self.index_keypoints = {x:i for i, x in enumerate(keypoints)}
        self.skelton   = skelton


    def read_json(self, json_path: str) -> pd.DataFrame:
        """
        CocoManager と互換性を取れるような DataFrame を作成する
        """
        logger.info(f"read json file: {json_path}")
        fjson = json.load(open(json_path))
        fname = self.dirpath_img + os.path.basename(fjson["imagePath"])
        img   = cv2.imread(fname)

        # labelme json dataframe 化する
        df = pd.DataFrame()
        df_json = pd.DataFrame(fjson["shapes"])
        for i, (label, points, shape_type) in enumerate(df_json[df_json["shape_type"].isin(["polygon","rectangle"])][["label","points","shape_type"]].values):
            if self.index_categories_name.get(label) is None: continue
            se = pd.Series(dtype=object)
            se["images_license"]   = 0
            se["images_file_name"] = os.path.basename(fjson["imagePath"])
            se["images_coco_url"]  = fname
            se["images_height"]    = img.shape[0]
            se["images_width"]     = img.shape[1]
            se["images_date_captured"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            se["images_flickr_url"] = None
            se["images_id"]               = 0 # dummy
            se["annotations_image_id"]    = 0 # dummy
            se["annotations_category_id"] = i
            ## bbox, segmentation
            if   shape_type == "rectangle":
                [[x1, y1], [x2, y2]] = points
                if x1 > x2:
                    x2 = points[0][0]
                    x1 = points[1][0]
                if y1 > y2:
                    y2 = points[0][1]
                    y1 = points[1][1]
                se["annotations_bbox"] = [int(x1), int(y1), int(x2-x1), int(y2-y1)]
                se["annotations_segmentation"] = []
                se["annotations_area"]    = 0
                se["annotations_iscrowd"] = 0
            elif shape_type == "polygon":
                ndf = np.zeros_like(img).astype(np.uint8)
                ndf = cv2.polylines(ndf, [np.array(points).reshape(-1,1,2).astype(np.int32)], True, (255,0,0))
                contours = cv2.findContours(ndf[:, :, 0], cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]
                x,y,w,h = cv2.boundingRect(contours[0])
                se["annotations_bbox"] = [int(x), int(y), int(w), int(h)]
                se["annotations_segmentation"] = [np.array(points).reshape(-1).astype(int).tolist()]
                se["annotations_area"] = cv2.contourArea(contours[0])
                se["annotations_iscrowd"] = 0
                x1, y1, x2, y2 = x, y, x+w, y+h               
            ## keypoint
            list_kpt  = self.keypoints_belong[label] if self.keypoints_belong is not None else self.keypoints # あるlabelが所属するkptを定義する
            dfwk      = df_json[(df_json["shape_type"] == "point") & (df_json["label"].isin(list_kpt))].copy()
            dfwk["x"] = dfwk["points"].map(lambda x: x[0][0])
            dfwk["y"] = dfwk["points"].map(lambda x: x[0][1])
            dfwk      = dfwk.loc[((dfwk["x"] >= x1) & (dfwk["x"] <= x2) & (dfwk["y"] >= y1) & (dfwk["y"] <= y2)), :]
            ndf = np.zeros(0)
            for keyname in self.keypoints:
                dfwkwk = dfwk[dfwk["label"] == keyname]
                if dfwkwk.shape[0] > 0:
                    sewk = dfwkwk.iloc[0]
                    ndf = np.append(ndf, (sewk["x"], sewk["y"], 2,))
                else:
                    ndf = np.append(ndf, (0, 0, 0,))
            se["annotations_keypoints"]     = ndf.reshape(-1).tolist()
            se["annotations_num_keypoints"] = (ndf.reshape(-1, 3)[:, -1] >= 1).sum() if ndf.shape[0] >= 3 else 0
            ## categories
            se["categories_id"]   = self.index_categories_name.get(label)
            se["categories_name"] = label
            se["categories_keypoints"] = self.keypoints if self.keypoints is not None else []
            se["categories_skeleton"]  = [[self.index_keypoints[x[0]], self.index_keypoints[x[1]]] for x in self.skelton] if self.skelton is not None else []
            df = df.append(se, ignore_index=True)
        return df


    def to_coco(self, save_path: str):
        df = pd.DataFrame()
        for x in get_file_list(self.dirpath_json, regex_list=[r"\.json"]):
            dfwk = self.read_json(x)
            df   = pd.concat([df, dfwk], ignore_index=True, sort=False, axis=0)
        coco = CocoManager()
        coco.df_json = df.copy()
        coco.re_index()
        coco.save(save_path)

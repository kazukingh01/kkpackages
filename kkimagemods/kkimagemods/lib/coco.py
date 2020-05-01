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
from kkimagemods.util.common import correct_dirpath, check_type, makedirs
from kkimagemods.util.logger import set_looger
logger = set_looger(__name__)

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
    """NDDS output convert to Coco format"""

    def __init__(self):
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
        self.ignore_jsons = ["_camera_settings.json", "_object_settings.json"]
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


    def set_base_parameter(
        self, dirpath: str, n_files: int = 100, instance_merge: dict = None, 
        visibility_threshold = 0.2, extra_pixel_x: int = 0, extra_pixel_y: int = 0, 
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
                    instance_merge={"human1":["body1","shoes1","clothes1", ...], "human2":["body2","shoes2","clothes2", ...], "dog1":["dog_body1", "dog_collars1"]}, 
                    visibility_threshold = 0.2
                    )
        """
        logger.debug("START")

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
            img_cs = cv2.cvtColor(cv2.imread(files[i].replace(".json", ".cs.png")), cv2.COLOR_BGR2GRAY)

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
                x2 = x2 if x2 <= img_cs.shape[1] else img_cs.shape[1]
                y2 = y2 if y2 <= img_cs.shape[0] else img_cs.shape[0]
                try:
                    listwk = img_cs[y1:y2, x1:x2].reshape(-1).tolist()
                except IndexError:
                    ## objectの中心が画面の外にはみ出してindex error になる場合があるのでその場合は省く
                    logger.warning("search range is out of images.")
                    continue
                if len(listwk) == 0: continue
                dfwk = pd.DataFrame(listwk, columns=["segmentation_color"])
                dfwk["file_name"]     = files[i]
                dfwk["category_name"] = dictwk["class"]

                ## 背景色も判断するために画面の端もsampleする.
                dfwk["segmentation_color_end"] = img_cs[0, 0]
                list_df.append(dfwk.copy())

            count_file += 1
            if n_files is not None and count_file > n_files: break
            
        # 解析
        df = pd.concat(list_df, ignore_index=True, sort=False, axis=0)
        for x in ["segmentation_color","segmentation_color_end"]: df[x] = df[x].astype(np.uint8)
        self.df_ndds = df.copy()
        ## 色のパターン
        dict_color = {x:None for x in df["category_name"].unique()}
        ## 背景色
        se = df["segmentation_color_end"].value_counts().sort_values(ascending=False)
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

        logger.debug("END")


    def __read_ndds_output(self, json_file_path: str):
        logger.debug("START")

        # image file
        image_path = json_file_path.replace(".json", ".png")
        image_name = json_file_path[json_file_path.rfind("/")+1:].replace(".json", ".png")

        img_cs = cv2.cvtColor(cv2.imread(json_file_path.replace(".json", ".cs.png")), cv2.COLOR_BGR2GRAY)
        height = img_cs.shape[0]
        width  = img_cs.shape[1]

        for name in self.instances.keys():
            # objects
            se = pd.Series(dtype=object)
            se["image_path"] = image_path
            se["image_name"] = image_name
            se["height"]     = height
            se["width"]      = width
            se["category_name"] = name

            ## instance の処理
            ndf = np.isin(img_cs, self.instances[name]).astype(np.uint8)
            ## 何も画像がない場合
            if np.sum(ndf) == 0: continue
            ## bounding box (自作) の処理
            x,y,w,h = cv2.boundingRect(ndf)
            se["bbox"] = (x,y,w,h,)
            ## segmmentation の処理
            contours = cv2.findContours(ndf, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]
            se["segmentation"] = [contour.reshape(-1) for contour in contours]

            # 画像情報を追加
            self.images = self.images.append(se, ignore_index=True)

        logger.debug("END")


    def read_ndds_output_all(self, dirpath: str):
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

        for x in self.__get_json_files(dirpath):
            logger.info(f"read file: {x}")
            self.__read_ndds_output(x)

        # 型の変換
        for x in ["height", "width"]:
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
    

    def sample_ouptut(self, n_images: int = 100, dirpath = "./sample_ouptut"):
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
        logger.debug("START")

        makedirs(dirpath, exist_ok=True, remake=True)
        ndf = self.images["image_name"].unique()
        for x in ndf[ np.random.permutation(np.arange(ndf.shape[0]))[:n_images] ]:
            img = self.draw_infomation(x)
            cv2.imwrite(dirpath + "/" + x, img)

        logger.debug("END")


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
                dict_category_id[y] = i # {"corn1": 0, "corn2": 0, ..., coneX: Y, ...}
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
                dictwk = {}
                dictwk["segmentation"]  = [_x.tolist() for _x in se["segmentation"]]
                dictwk["area"]          = sum([cv2.contourArea(_x.reshape(-1, 1, 2)) for _x in se["segmentation"]])
                dictwk["bbox"]          = list(se["bbox"])
                dictwk["iscrowd"]       = 0
                dictwk["image_id"]      = i
                dictwk["category_id"]   = int(dict_category_id[se["category_name"]]) # np.int32 形式だとjson変換でエラーになるため
                dictwk["id"]            = int(df.index[i_index]) # とりあえずself.images の index が一意なのでそれをつける
                dictwk["num_keypoints"] = 0
                dictwk["keypoints"]     = []
                json_dict["annotations"].append(dictwk)
        
        # categories
        json_dict["categories"] = []
        for x in category_merge["supercategory"].keys():
            for y in category_merge["supercategory"][x]:
                json_dict["categories"].append({"supercategory":x, "name":y, "id": dict_category_id_name[y], "keypoints":[] })

        logger.debug("END")
        return json.dumps(json_dict)
    

    def output_coco_format(self, dirpath: str, category_merge: dict):
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

        _dirpath = correct_dirpath(dirpath)
        os.makedirs(_dirpath + "json",   exist_ok=True)
        os.makedirs(_dirpath + "images", exist_ok=True)

        # json output
        with open(_dirpath + "json/instances.json", mode="w") as f:
            f.write(self.to_coco_format(category_merge=category_merge))
        
        # image copy
        for x in self.images["image_path"].unique():
            shutil.copy2(x, _dirpath + "images/")

        logger.debug("END")



class CocoManager:
    """Coco format manager with pandas DataFrame"""

    def __init__(self):
        self.df_json   = pd.DataFrame()
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
            raise Exception(f"same file name: [{se[se.apply(lambda x: len(x) > 1)].index.values}]")


    def re_index(self):
        # image id
        dictwk = {x:i  for i, x in enumerate(np.sort(self.df_json["images_coco_url"].unique()))}
        self.df_json["images_id"]            = self.df_json["images_coco_url"].map(dictwk)
        self.df_json["annotations_image_id"] = self.df_json["images_coco_url"].map(dictwk)
        # license id
        dictwk = {x:i  for i, x in enumerate(np.sort(self.df_json["licenses_name"].unique()))}
        self.df_json["images_license"] = self.df_json["licenses_name"].map(dictwk)
        self.df_json["licenses_id"]    = self.df_json["licenses_name"].map(dictwk)
        # category id ( super category と category name で一意とする )
        self.df_json["__work"] = (self.df_json["categories_supercategory"].astype(str) + "_" + self.df_json["categories_name"].astype(str)).copy()
        dictwk = {x:i  for i, x in enumerate(np.sort(self.df_json["__work"].unique()))}
        self.df_json["annotations_category_id"] = self.df_json["__work"].map(dictwk)
        self.df_json["categories_id"]           = self.df_json["__work"].map(dictwk)
        self.df_json = self.df_json.drop(columns=["__work"])
        # annotations id ( df の index に相当する)
        self.df_json = self.df_json.reset_index(drop=True)
        self.df_json["annotations_id"] = self.df_json.index.values


    def add_json(self, src):
        check_type(src, [str, dict])
        json_coco = {}
        if   type(src) == str:  json_coco = json.load(open(src))
        elif type(src) == dict: json_coco = src
        try:
            self.coco_info = json_coco["info"]
        except KeyError:
            print("'src' file or dictionary is not found 'info' key. so, skip this src.")
            return None
        df = self.json_to_df(src)
        self.df_json = pd.concat([self.df_json, df], axis=0, ignore_index=True, sort=False)
        self.check_index()
        self.re_index()
    

    def to_coco_format(self):
        json_dict = {}
        json_dict["info"]   = self.coco_info
        for _name in ["images", "annotations", "licenses", "categories"]:
            df = self.df_json.loc[:, self.df_json.columns.str.contains("^"+_name+"_", regex=True)].copy()
            df.columns = df.columns.str[len(_name+"_"):]
            df = df.fillna("%%null%%")
            json_dict[_name] = df.groupby("id").first().reset_index().apply(lambda x: x.to_dict(), axis=1).to_list()
        strjson = json.dumps(json_dict)
        return strjson.replace('"%%null%%"', 'null')


    def save(self, filename: str, save_images_path: str = None, exist_ok = False, remake = False):
        if save_images_path is not None:
            save_images_path = correct_dirpath(save_images_path)
            makedirs(save_images_path, exist_ok = exist_ok, remake = remake)
            # image copy
            for x in self.df_json["images_coco_url"].unique():
                shutil.copy2(x, save_images_path)

        with open(filename, "w") as f:
            f.write(self.to_coco_format())


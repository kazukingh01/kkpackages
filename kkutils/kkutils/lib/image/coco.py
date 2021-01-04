import json, os, shutil, datetime
import pandas as pd
import numpy as np
import cv2
from typing import List
# local package
from kkutils.util.image import coco_info
from kkutils.util.com import check_type, makedirs, get_file_list, correct_dirpath, set_logger, get_filename
logger = set_logger(__name__)


__all__ = [
    "CocoManager"
]


class CocoManager:
    """Coco format manager with pandas DataFrame"""

    def __init__(self):
        self.df_json   = pd.DataFrame()
        self.json      = {} # 直近の１つを保存する
        self.coco_info = {}
        self._dict_imgpath = {}
        self._dict_ann     = {}
        self._dict_cat     = {}
        self._list_se      = []
        self.initialize()


    def initialize(self):
        """
        images_id                                                                    0
        images_coco_url                                       ./traindata/000000.0.png
        images_date_captured                                       2020-07-30 16:07:12
        images_file_name                                                  000000.0.png
        images_flickr_url                                                  http://test
        images_height                                                             1400
        images_license                                                               0
        images_width                                                              1400
        annotations_id                                                               0
        annotations_area                                                         11771
        annotations_bbox             [727.0316772460938, 596.92626953125, 124.13433...
        annotations_category_id                                                      0
        annotations_image_id                                                         0
        annotations_iscrowd                                                          0
        annotations_keypoints        [834, 855, 2, 821, 649, 2, 780, 615, 2, 750, 6...
        annotations_num_keypoints                                                    5
        annotations_segmentation     [[773, 602, 772, 603, 771, 603, 770, 603, 769,...
        licenses_id                                                                  0
        licenses_name                                                     test license
        licenses_url                                                       http://test
        categories_id                                                                0
        categories_keypoints         [kpt_a, kpt_cb, kpt_c, kpt_cd, kpt_e, kpt_b, k...
        categories_name                                                           hook
        categories_skeleton                   [[0, 1], [1, 2], [2, 3], [3, 4], [5, 6]]
        categories_supercategory                                                  hook
        """
        self.df_json = pd.DataFrame(
            columns=[
                'images_id', 'images_coco_url', 'images_date_captured',
                'images_file_name', 'images_flickr_url', 'images_height',
                'images_license', 'images_width', 'annotations_id', 'annotations_area',
                'annotations_bbox', 'annotations_category_id', 'annotations_image_id',
                'annotations_iscrowd', 'annotations_keypoints',
                'annotations_num_keypoints', 'annotations_segmentation', 'licenses_id',
                'licenses_name', 'licenses_url', 'categories_id',
                'categories_keypoints', 'categories_name', 'categories_skeleton',
                'categories_supercategory'
            ]
        )
        self.coco_info = coco_info()


    def add(
        self, imgpath: str, height: int, width: int, 
        bbox: (float, float, float, float), 
        category_name: str, super_category_name: str=None,
        segmentations: List[List[float]]=None,
        keypoints: List[float]=None,
        category_name_kpts: List[str]=None,
    ):
        """
        add で self._list_se に貯める. 毎回 df.append すると遅いので
        Params::
            imgpath: image path
            bbox: [x_min, y_min, width, height]
            category_name: class name
            super_category_name: if you want to define, set super category name. default is None
            segmentations: [[x11, y11, x12, y12, ...], [x21, y21, x22, y22, ...], ..]
            keypoints: [x1, y1, vis1, x2, y2, vis2, ...]
            category_name_kpts: ["kpt1", "kpt2", ...]
        """
        if imgpath       not in self._dict_imgpath: self._dict_imgpath[imgpath]   = len(self._dict_imgpath)
        if category_name not in self._dict_cat:     self._dict_cat[category_name] = len(self._dict_cat)
        self._dict_ann[imgpath] = (self._dict_ann[imgpath] + 1) if imgpath in self._dict_ann else 0
        se = pd.Series()
        se["images_id"] = self._dict_imgpath[imgpath]
        se["images_file_name"] = os.path.basename(imgpath)
        se["images_coco_url"] = imgpath
        se["images_date_captured"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        se["images_flickr_url"] = "http://test"
        se["images_height"] = height
        se["images_width"] = width
        se["images_license"] = 0
        se["annotations_id"] = self._dict_ann[imgpath]
        se["annotations_bbox"] = list(bbox)
        se["annotations_area"] = int(se["annotations_bbox"][-2] * se["annotations_bbox"][-1])
        se["annotations_category_id"] = self._dict_cat[category_name]
        se["annotations_image_id"] = self._dict_imgpath[imgpath]
        se["annotations_iscrowd"] = 0
        se["annotations_keypoints"] = keypoints if keypoints is not None else []
        se["annotations_num_keypoints"] = len(keypoints)//3 if keypoints is not None else 0
        se["annotations_segmentation"] = segmentations if segmentations is not None else []
        se["licenses_id"] = 0
        se["licenses_name"] = "test license"
        se["licenses_url"] = "http://test"
        se["categories_id"] = self._dict_cat[category_name]
        se["categories_keypoints"] = category_name_kpts if category_name_kpts is not None else []
        se["categories_name"] = category_name
        se["categories_skeleton"] = []
        se["categories_supercategory"] = category_name if super_category_name is None else super_category_name
        self._list_se.append(se)
    

    def concat_added(self):
        self.df_json = pd.concat(self._list_se, ignore_index=True, sort=False, axis=1).T
        self.re_index()


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
            ["categories_keypoints", "categories_skeleton", "licenses_name", "licenses_url", "annotations_segmentation", "annotations_keypoints"], 
            [[], [], np.nan, np.nan, [], []]
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
            logger.warning("'src' file or dictionary is not found 'info' key. so, skip this src.")
            self.coco_info = coco_info()
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
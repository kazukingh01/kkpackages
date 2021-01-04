import os, datetime, copy, time
import numpy as np
import pandas as pd
import cv2
from typing import List, Tuple
import torch

from kkimgaug.lib.aug_det2 import Mapper

# detectron2 packages
from detectron2.engine import DefaultTrainer, DefaultPredictor, HookBase
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import build_detection_train_loader, DatasetCatalog, MetadataCatalog
from detectron2.data.dataset_mapper import DatasetMapper
import detectron2.data.detection_utils as utils
import detectron2.data.transforms as T
from fvcore.common.config import CfgNode
import detectron2.utils.comm as comm

# local package
from kkimagemods.util.common import makedirs, correct_dirpath
from kkimagemods.lib.coco import coco_info, CocoManager
from kkimagemods.util.images import drow_bboxes, convert_seg_point_to_bool, fit_resize


class MyDet2(DefaultTrainer):
    def __init__(
            self,
            # coco dataset
            dataset_name: str = None, coco_json_path: str=None, image_root: str=None,
            # train params
            cfg: CfgNode=None, max_iter: int=100, is_train: bool=True, aug_json_file_path: str=None, 
            base_lr: float=0.01, batch_size: int=1, num_workers: int=3, resume: bool=False, 
            lr_steps: (int,int,)=None, lr_warmup: int=1000, 
            # validation param
            validations: List[Tuple[str]]=None, valid_steps: int=100, valid_ndata: int=10,
            # train and test params
            model_zoo_path: str="COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml", weight_path: str=None, 
            is_keyseg: bool=False, is_bbox_only: bool=False, save_step: int=5000,
            classes: List[str] = None, keypoint_names: List[str] = None, keypoint_flip_map: List[Tuple[str]] = None,
            input_size: tuple=((640, 672, 704, 736, 768, 800), 1333), threshold: float=0.2, outdir: str="./output"
        ):
        # coco dataset
        self.dataset_name       = dataset_name
        self.model_zoo_path     = model_zoo_path
        self.coco_json_path     = coco_json_path
        self.coco_json_path_org = coco_json_path
        self.image_root         = image_root
        self.is_train           = is_train
        if (self.dataset_name is None or self.coco_json_path is None or self.image_root is None) == False:
            self.__register_coco_instances(self.dataset_name, self.coco_json_path, self.image_root) # Coco dataset setting
        self.cfg                = cfg if cfg is not None else self.set_config(
            weight_path=weight_path, threshold=threshold, max_iter=max_iter, num_workers=num_workers, 
            batch_size=batch_size, base_lr=base_lr, lr_steps=lr_steps, input_size=input_size, outdir=outdir
        )
        # cfg に対する追加の設定
        if is_keyseg:
            self.cfg.MODEL.MASK_ON     = True
            self.cfg.MODEL.KEYPOINT_ON = True
        if is_bbox_only:
            self.cfg.MODEL.MASK_ON     = False
            self.cfg.MODEL.KEYPOINT_ON = False
        self.cfg.INPUT.RANDOM_FLIP = "none"
        self.cfg.SOLVER.CHECKPOINT_PERIOD = save_step
        self.cfg.SOLVER.WARMUP_FACTOR = 1.0 / lr_warmup
        self.cfg.SOLVER.WARMUP_ITERS = lr_warmup
        self.cfg.SOLVER.WARMUP_METHOD = "linear"
        # classes は強制でセットする
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)
        MetadataCatalog.get(self.dataset_name).thing_classes = classes
        if keypoint_names is not None:
            self.cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = len(keypoint_names)
            # Key point の metadata を set
            MetadataCatalog.get(self.dataset_name).keypoint_names = keypoint_names
            MetadataCatalog.get(self.dataset_name).keypoint_flip_map = keypoint_flip_map
            MetadataCatalog.get(self.dataset_name).keypoint_connection_rules = [(x[0], x[1], (255,0,0)) for x in keypoint_flip_map] # Visualizer の内部で使用している
        self.mapper = None if aug_json_file_path is None else Mapper(self.cfg, config=aug_json_file_path)

        if self.is_train:
            # train setting
            os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True) # この宣言は先にする
            super().__init__(self.cfg) # train の時しか init しない
            self.predictor = None
            self.resume_or_load(resume=resume) # Falseだとload the model specified by the config (skip all checkpointables).

            # validation setting
            if validations is not None:
                list_validator: list = []
                for valid in validations:
                    self.__register_coco_instances(*valid) # valid: (dataset_name, json_path, image_path)
                    MetadataCatalog.get(valid[0]).thing_classes = classes
                    if keypoint_names is not None:
                        self.cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = len(keypoint_names)
                        # Key point の metadata を set
                        MetadataCatalog.get(valid[0]).keypoint_names = keypoint_names
                        MetadataCatalog.get(valid[0]).keypoint_flip_map = keypoint_flip_map
                        MetadataCatalog.get(valid[0]).keypoint_connection_rules = [(x[0], x[1], (255,0,0)) for x in keypoint_flip_map] # Visualizer の内部で使用している
                    list_validator.append(Validator(self.cfg.clone(), valid[0], trainer=self, steps=valid_steps, ndata=valid_ndata))
                self.register_hooks(list_validator) # Hook の登録. after_stepがtrain中に呼び出される
 
        # この定義は最後がいいかも
        if self.is_train == False:
            # test setting
            self.set_predictor()


    @classmethod
    def __register_coco_instances(cls, dataset_name: str, coco_json_path: str, image_root: str):
         # この関数で内部のDatasetCatalog, MetadataCatalogにCoco情報をset している
        register_coco_instances(dataset_name, {}, coco_json_path, image_root)


    # override. super().__init__ 内でこの関数が呼ばれる
    def build_train_loader(self, cfg) -> torch.utils.data.DataLoader:
        return build_detection_train_loader(cfg, mapper=self.mapper)


    def set_config(
        self, weight_path: str=None, threshold: float=0.2, max_iter: int=100, num_workers: int=2, 
        batch_size: int=1, base_lr: float=0.01, lr_steps: (int,int)=None, input_size: tuple=(800,1333,), outdir: str="./output"
    ) -> CfgNode:
        """
        see https://detectron2.readthedocs.io/modules/config.html#detectron2.config.CfgNode
        """
        # common setting
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(self.model_zoo_path))
        cfg.OUTPUT_DIR = outdir
        cfg.DATASETS.TRAIN = (self.dataset_name, ) # DatasetCatalog, MetadataCatalog の中で自分でsetした"my_dataset_train"を指定
        cfg.DATASETS.TEST  = (self.dataset_name, )
        cfg.DATALOADER.NUM_WORKERS = num_workers
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.model_zoo_path) if weight_path is None else weight_path
        cfg.SOLVER.IMS_PER_BATCH = batch_size
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512 # faster, and good enough for this toy dataset (default: 512)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold if threshold is not None else 0.2  # set the testing threshold for this model
        cfg.INPUT.MIN_SIZE_TRAIN = input_size[0]
        cfg.INPUT.MAX_SIZE_TRAIN = input_size[1]
        cfg.INPUT.MIN_SIZE_TEST  = input_size[0]
        cfg.INPUT.MAX_SIZE_TEST  = input_size[1]
        if self.is_train:
            cfg.SOLVER.BASE_LR = base_lr # pick a good LR
            cfg.SOLVER.MAX_ITER = max_iter    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
            if lr_steps is not None:
                cfg.SOLVER.STEPS = tuple(lr_steps)
        return cfg


    def train(self):
        makedirs(self.cfg.OUTPUT_DIR, exist_ok=True, remake=False)
        super().train()
        self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, "model_final.pth")
        self.set_predictor()


    def set_predictor(self):
        self.predictor = DefaultPredictor(self.cfg)


    def predict(self, data: np.ndarray):
        if self.predictor is None:
            self.set_predictor()
        return self.predictor(data)


    def img_crop_bbox(self, img: np.ndarray, padding: int=0, _class=None) -> List[np.ndarray]:
        """
        Params::
            _class: str or int. str の場合はそのクラス名に一致するものを全て切り抜く。intはそのindexのみ。Noneは全部切り抜く
        """
        output_list = []
        # 推論
        output = self.predictor(img)
        output = output["instances"]
        ndf     = output.get("pred_boxes").to("cpu").tensor.numpy().copy()
        classes = output.get("pred_classes").to("cpu").numpy().copy()
        if _class is None:
            for x1, y1, x2, y2 in ndf:
                _y1, _y2, _x1, _x2 = int(y1)-padding, int(y2)+padding, int(x1)-padding, int(x2)+padding
                _y1 = _y1 if _y1 >= 0 else 0
                _x1 = _x1 if _x1 >= 0 else 0
                imgwk = img[_y1:_y2, _x1:_x2, :].copy()
                output_list.append(imgwk)
        elif isinstance(_class, list) or isinstance(_class, str):
            if isinstance(_class, str): _class = [_class]
            for i, index in enumerate(classes):
                if MetadataCatalog.get(self.dataset_name).thing_classes[index] in _class:
                    x1, y1, x2, y2 = ndf[i]
                    _y1, _y2, _x1, _x2 = int(y1)-padding, int(y2)+padding, int(x1)-padding, int(x2)+padding
                    _y1 = _y1 if _y1 >= 0 else 0
                    _x1 = _x1 if _x1 >= 0 else 0
                    imgwk = img[_y1:_y2, _x1:_x2, :].copy()
                    output_list.append(imgwk)
        elif isinstance(_class, int):
            x1, y1, x2, y2 = ndf[_class]
            _y1, _y2, _x1, _x2 = int(y1)-padding, int(y2)+padding, int(x1)-padding, int(x2)+padding
            _y1 = _y1 if _y1 >= 0 else 0
            _x1 = _x1 if _x1 >= 0 else 0
            imgwk = img[_y1:_y2, _x1:_x2, :].copy()
            output_list.append(imgwk)
        return output_list


    def show(self, img: np.ndarray, add_padding: int=0, only_best: bool = False, preview: bool=False, resize: bool=False) -> np.ndarray:
        output = self.predict(img)
        for i in range(output["instances"].get("pred_boxes").tensor.shape[0]):
            for j in range(4):
                output["instances"].get("pred_boxes").tensor[i][j] += add_padding
        # padding. annotation が見えなくなる場合もあるため
        if add_padding > 0:
            img = cv2.copyMakeBorder(
                img, add_padding, add_padding, add_padding, add_padding,
                cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
        if only_best:
            output["instances"] = output["instances"][0:1]
        img_ret = self.draw_annoetation(img, output)
        if resize:
            img_ret = fit_resize(img_ret, "y", 1000)
        if preview:
            cv2.imshow(__name__, img_ret)
            cv2.waitKey(0)
        return img_ret


    def draw_annoetation(self, img: np.ndarray, data: dict):
        from detectron2.data import MetadataCatalog
        import detectron2.utils.visualizer
        detectron2.utils.visualizer._KEYPOINT_THRESHOLD = 0
        metadata = MetadataCatalog.get(self.dataset_name)
        v = Visualizer(
            img[:, :, ::-1],
            metadata=metadata, 
            scale=1.0, 
            instance_mode=ColorMode.IMAGE #ColorMode.IMAGE_BW # remove the colors of unsegmented pixels
        )
        v = v.draw_instance_predictions(data["instances"].to("cpu"))
        img_ret = v.get_image()[:, :, ::-1]
        return img_ret


    def __to_coco(self, img_path: str):
        # 画像の読み込み
        img = cv2.imread(img_path)
        # 推論
        output = self.predict(img)
        output = output["instances"]
        # coco format
        json_dict = {}
        json_dict["info"] = coco_info()
        # licenses
        json_dict["licenses"] = []
        json_dict["licenses"].append({"url":"http://test","id":0,"name":"test license"})
        # images
        json_dict["images"] = []
        dictwk = {}
        dictwk["license"]       = 0
        dictwk["file_name"]     = os.path.basename(img_path)
        dictwk["coco_url"]      = img_path
        dictwk["height"]        = int(img.shape[0]) # np.int32 形式だとjson変換でエラーになるため
        dictwk["width"]         = int(img.shape[1]) # np.int32 形式だとjson変換でエラーになるため
        dictwk["date_captured"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        dictwk["flickr_url"]    = "http://test"
        dictwk["id"]            = 0
        json_dict["images"].append(dictwk)
        # annotations
        json_dict["annotations"] = []
        ## segmentation, bbox で処理が変わる. segmentation があればそっちを優先する
        ndf, is_segmentation = None, True
        try:
            ndf = output.get("pred_masks").to("cpu").detach().numpy().copy()
        except KeyError:
            is_segmentation = False
            ndf = output.get("pred_boxes").to("cpu").tensor.numpy().copy() # detach できないので気をつける
        ndf_cat = output.get("pred_classes").to("cpu").detach().numpy().copy()
        ndf_sco = output.get("scores")      .to("cpu").detach().numpy().copy()
        for i_index in np.arange(ndf.shape[0]):
            if ndf_sco[i_index] < self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST: continue # threshold より低いscoreは見ない
            dictwk = {}
            ## segmentation
            if is_segmentation:
                contours = cv2.findContours(ndf[i_index, ::].astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]
                segmentation = [contour.reshape(-1) for contour in contours]
                dictwk["segmentation"]  = [_x.tolist() for _x in segmentation]
                dictwk["area"]          = sum([cv2.contourArea(_x.reshape(-1, 1, 2)) for _x in segmentation])
            else:
                dictwk["segmentation"] = []
                dictwk["area"]         = 0
            ## bounding box
            if is_segmentation:
                x,y,w,h = cv2.boundingRect(ndf[i_index, ::].astype(np.uint8))
            else:
                x1, y1, x2, y2 = ndf[i_index]
                x = int(x1)
                y = int(y1)
                w = int(x2 - x1)
                h = int(y2 - y1)
            dictwk["bbox"]          = [x,y,w,h,]
            dictwk["iscrowd"]       = 0
            dictwk["image_id"]      = 0
            dictwk["category_id"]   = int(ndf_cat[i_index]) # np.int32 形式だとjson変換でエラーになるため
            dictwk["id"]            = int(i_index)
            dictwk["num_keypoints"] = 0
            dictwk["keypoints"]     = []
            json_dict["annotations"].append(dictwk)
        if len(json_dict["annotations"]) == 0:
            # annotation がない場合は学習するデータがないので、coco json をそもそも作らない
            return {}

        # categories
        json_dict["categories"] = []
        for x in np.sort(np.unique(ndf_cat)):
            json_dict["categories"].append({"supercategory":x, "name":x, "id": x, "keypoints":[], "skeleton":[], })
        
        return json_dict


    def to_coco(self, img_paths: List[str], out_filename: str, dict_categories={"supercategory":None, "categories":None}):
        coco_manager = CocoManager()
        for x in img_paths:
            print(x)
            json_coco = self.__to_coco(x)
            coco_manager.add_json(json_coco)
        if dict_categories["supercategory"] is not None:
            coco_manager.df_json["categories_supercategory"] = coco_manager.df_json["categories_supercategory"].map(dict_categories["supercategory"])
        if dict_categories["categories"] is not None:
            coco_manager.df_json["categories_name"] = coco_manager.df_json["categories_id"].map(dict_categories["categories"])
        coco_manager.save(out_filename)


    @classmethod
    def img_conv_dataloader(cls, data):
        """
        dataloader で読み出した画像を解釈するためのクラス
        """
        img = data["image"].detach().numpy().copy().T.astype(np.uint8)
        img = np.rot90(img, 1)
        img = np.flipud(img)
        return img


    def preview_augmentation(self, src, outdir: str="./preview_augmentation", n_output: int=100):
        """
        面倒なのでcocoを作り直してからpreviewさせる
        Params::
            src: str, List[str], index, List[index]
        """
        outdir = correct_dirpath(outdir)
        coco   = CocoManager()
        coco.add_json(self.coco_json_path)
        # src で絞る
        if   type(src) == str:
            coco.df_json = coco.df_json.loc[coco.df_json["images_file_name"] == src]
        elif type(src) == int:
            coco.df_json = coco.df_json.iloc[src:src+1]
        elif type(src) == list or type(src) == tuple:
            if   type(src[0]) == str:
                coco.df_json = coco.df_json.loc[coco.df_json["images_file_name"].isin(src)]
            elif type(src[0]) == int:
                coco.df_json = coco.df_json.iloc[src, :]
        else:
            raise Exception("")
        coco.save(self.coco_json_path + ".cocomanager.json")

        # 作り直したcocoで再度読み込みさせる
        self.coco_json_path = self.coco_json_path + ".cocomanager.json"
        DatasetCatalog.remove(self.dataset_name)  # key を削除しないと再登録できない
        MetadataCatalog.remove(self.dataset_name) # key を削除しないと再登録できない
        self.__register_coco_instances(self.dataset_name, self.coco_json_path, self.image_root)
        super().__init__(self.cfg)
        makedirs(outdir, exist_ok=True, remake=True)
        count = 0
        for i, x in enumerate(self.data_loader):
            # x には per batch 分の size (2個とか) 入っているので、それ分回す
            for j, data in enumerate(x):
                if j > 0: continue
                ## Visualizer を predictor と統一するため, gt_*** -> pred_*** に copy する
                img = self.img_conv_dataloader(data)
                ins = data["instances"].to("cpu")
                if ins.has("gt_boxes"):     ins.set("pred_boxes",     ins.gt_boxes)
                if ins.has("gt_classes"):   ins.set("pred_classes",   ins.gt_classes)
                if ins.has("gt_keypoints"): ins.set("pred_keypoints", ins.gt_keypoints)
                if ins.has("gt_masks"):
                    ## gt_mask では [x1, y1, x2, y2, ... ]の形式になっているのでそれを pred [False, True, True, ...] 形式に変換する
                    segs = ins.get("gt_masks").polygons
                    list_ndf = []
                    for seg_a_class in segs:
                        ndf = convert_seg_point_to_bool(img.shape[0], img.shape[1], seg_a_class)
                        list_ndf.append(ndf)
                    ndf = np.concatenate([[ndfwk] for ndfwk in list_ndf], axis=0)
                    ins.set("pred_masks", torch.from_numpy(ndf))  # Tensor 形式に変換
                data["instances"] = ins
                img = self.draw_annoetation(img, data)
                cv2.imwrite(outdir + "preview_augmentation." + str(i) + "." + str(j) + ".png", img)
            count += 1
            if count > n_output: break

        DatasetCatalog.remove(self.dataset_name)  # key を削除しないと再登録できない
        MetadataCatalog.remove(self.dataset_name) # key を削除しないと再登録できない
        self.coco_json_path = self.coco_json_path_org
        self.__register_coco_instances(self.dataset_name, self.coco_json_path, self.image_root)
        super().__init__(self.cfg)
    

    def predict_to_df(self, img: np.ndarray) -> pd.DataFrame:
        from detectron2.data import MetadataCatalog
        metadata = MetadataCatalog.get(self.dataset_name)
        output = self.predict(img)
        output = output["instances"].to("cpu")
        df = pd.DataFrame()
        for i, x in enumerate(output.get("pred_classes")):
            se = pd.Series(dtype=object)
            se["class_name"] = metadata.thing_classes[x]
            se["bbox"]       = output.get("pred_boxes").tensor[i].detach().numpy()
            se["bbox_score"] = float(output.get("scores")[i])
            df = df.append(se, ignore_index=True)
        return df
    

    @classmethod
    def __predict_to_df(cls, output: dict) -> pd.DataFrame:
        output = output["instances"].to("cpu")
        df = pd.DataFrame(output.get("pred_boxes").tensor.detach().numpy(), columns=["x1","y1","x2","y2"], dtype=object)
        df["score"] = output.get("scores").detach().numpy()
        df["class"] = output.get("pred_classes").detach().numpy()
        df["keypoints"] = np.nan # ndarray 参照形式で代入しようとしたがobject typeが複数ある場合、values で ndf の copy が作成されるため参照形式で修正できない
        try:
            ndf = df.values
            ndf_key = output.get("pred_keypoints").detach().numpy()
            if ndf_key.shape[0] > 0:
                ndf[:, -1] = ndf_key.reshape(df.shape[0], -1).tolist()
                df = pd.DataFrame(ndf, columns=df.columns)
        except KeyError:
            # keypoint がない場合は pass
            pass
        return df
    

    def eval_a_image(
        self, imgpath: str, coco_gt: CocoManager, 
        classes: List[str], thre_iou: float=0.5,
        keypoints: List[str]=None, skeleton: List[List[str]]=None
    ) -> pd.DataFrame:
        df_json = coco_gt.df_json
        img     = cv2.imread(imgpath)
        output  = self.predict(img)
        fname   = os.path.basename(imgpath)
        df      = self.__predict_to_df(output)
        df["file_name"] = fname
        df["class"]     = df["class"].map({i:x for i, x in enumerate(classes)}) 
        # IoU
        ndf_base = np.zeros_like(img[:, :, 0]).astype(bool)
        df_anno = df_json[df_json["images_file_name"] == fname]
        for index, (class_name, [x,y,w,h]) in enumerate(df_anno[["categories_name","annotations_bbox"]].values):
            df["gt_bbox_"+class_name+"_"+str(index)] = df["class"].apply(lambda _: [x,y,w,h]) # class の箇所はなんでも良い
            colname = "gt_iou_"+class_name+"_"+str(index)
            ndf_gt = ndf_base.copy()
            ndf_gt[int(y):int(y+h+1), int(x):int(x+w+1)] = True
            df[colname] = np.nan
            for i, (x1, y1, x2, y2, ) in enumerate(df[["x1","y1","x2","y2"]].values):
                ndf_pred = ndf_base.copy()
                ndf_pred[int(y1):int(y2+1), int(x1):int(x2+1)] = True
                iou = (ndf_gt & ndf_pred).sum() / (ndf_gt | ndf_pred).sum()
                df.loc[df.index[i], colname] = iou
            # 予測classと違うclass のiou は nan にしておく
            df.loc[df["class"] != class_name, colname] = np.nan
        # tp, fp
        for class_name in classes:
            df["gt_"+class_name+"_n"] = df.columns.str.contains("gt_iou_"+class_name).sum()
            for y in df.columns[df.columns.str.contains("^gt_iou_"+class_name, regex=True)]:
                boolwk = (df["class"] == class_name)
                df.loc[~boolwk, y] = np.nan # 別class の predictで IoU を計算している値はnan に変換
                colname = y.replace("gt_iou_","pred_")
                ## pos:: 3: tp, 2: fp1(IoU閾値を満たしていない) 1: fp2(IoU が全く無い)
                df[colname] = np.nan
                df.loc[(df[y] >= thre_iou), colname] = 3.0
                df.loc[(df[y] <  thre_iou), colname] = 2.0
                df.loc[(df[y] == 0), colname] = 1.0
            # ある prediction が tp か fp1 か fp2 なのかを分類する
            dfwk = df.loc[:, df.columns.str.contains("^pred_"+class_name+"_[0-9]+$", regex=True)].copy()
            df["pred_"+class_name] = np.nan
            dfwkwk = df.loc[:, df.columns.str.contains("^gt_iou_"+class_name, regex=True)].copy().astype(float)
            if dfwkwk.shape[0] > 0 and dfwkwk.shape[1] > 0:
                dfwkwk = dfwkwk.loc[~dfwkwk.iloc[:, 0].isna(), :]
                sewkwk = dfwkwk.idxmax(axis=1).apply(lambda x: x.split("_")[-1]).astype(int)
                df.loc[dfwk[dfwk.max(axis=1) == 3].index.values, ("pred_"+class_name)] = sewkwk.loc[dfwk[(dfwk.max(axis=1) == 3)].index.values] # TPの場合は、どのGTに対してのTPかのラベルを残す
                df.loc[dfwk[dfwk.max(axis=1) == 2].index.values, ("pred_"+class_name)] = -1
                df.loc[dfwk[dfwk.max(axis=1) == 1].index.values, ("pred_"+class_name)] = -1
                # keypoint(tp のデータについて調査)
                if keypoints is not None:
                    keypoints = np.array(keypoints)
                    instances = df["pred_"+class_name].unique()
                    instances = instances[instances >= 0].astype(int)
                    df["gt_keys_"  +class_name] = np.nan
                    df["pred_keys_"+class_name] = np.nan
                    df["pred_sklt_"+class_name] = np.nan
                    for i_gt in instances: # i_gtは df_anno の index 番号と一致する
                        ## ここから gt のループ
                        se_anno   = df_anno.iloc[i_gt]
                        bool_keys = np.isin(se_anno["categories_keypoints"], keypoints)
                        gt_keys   = np.array(se_anno["annotations_keypoints"]).reshape(-1, 3)
                        gt_keys   = gt_keys[bool_keys, :]
                        gt_len_base = np.sqrt(se_anno["annotations_bbox"][2] ** 2 + se_anno["annotations_bbox"][3] ** 2)
                        bool_loc  = (df["pred_"+class_name] == i_gt).values
                        df.loc[bool_loc, "gt_keys_"+class_name] = df.loc[bool_loc, "class"].apply(lambda _: gt_keys.reshape(-1).tolist()) # class の箇所はなんでも良い
                        ndf = df.values.copy() # list を代入できないので、ndf で置き換えてから、再度dfを作成する
                        for i_pred in df.index[bool_loc]:
                            se = df.loc[i_pred, :].copy()
                            pred_keys = np.array(se["keypoints"]).reshape(-1, 3)
                            diff_keys = np.sqrt((gt_keys[:, 0] - pred_keys[:, 0]) ** 2 + (gt_keys[:, 1] - pred_keys[:, 1]) ** 2)
                            diff_keys = diff_keys / gt_len_base # bbox の対角線の長さで規格化する
                            diff_keys[gt_keys[:, -1] == 0] = -1 # gt が無い keypoint の誤差は -1 とする
                            diff_keys = np.append(diff_keys, pred_keys[:, -1]).reshape(2, -1).T.reshape(-1).tolist() # keypointのscoreを追加しておく
                            ndf[i_pred, -2] = diff_keys # ndarray形式でlistを代入. i_predは indexと一致しているはず
                            ## skelton の処理. gt とのvector の差を出しておく (diff_x, diff_y, ratio_length)
                            if skeleton is not None:
                                list_sklt = []
                                for [key1, key2] in skeleton:
                                    gt_key1   = gt_keys[  keypoints == key1, :].reshape(-1)
                                    gt_key2   = gt_keys[  keypoints == key2, :].reshape(-1)
                                    pred_key1 = pred_keys[keypoints == key1, :].reshape(-1)
                                    pred_key2 = pred_keys[keypoints == key2, :].reshape(-1)
                                    vec_gt    = [gt_key2[  0] - gt_key1[  0], gt_key2[  1] - gt_key1[  1]] # x,y
                                    vec_pred  = [pred_key2[0] - pred_key1[0], pred_key2[1] - pred_key1[1]] # x,y
                                    list_sklt.append(
                                        [
                                            (vec_pred[0] - vec_gt[0])/gt_len_base,
                                            (vec_pred[1] - vec_gt[1])/gt_len_base, 
                                            np.sqrt(vec_pred[0]**2 + vec_pred[1]**2) / np.sqrt(vec_gt[0]**2 + vec_gt[1]**2)
                                        ]
                                    )
                                ndf[i_pred, -1] = list_sklt
                        df = pd.DataFrame(ndf, columns=df.columns)
        return df
    

    def eval_images(
        self, imgdir: str, coco_gt: CocoManager, 
        classes: List[str]=None, thre_iou: float=0.5,
        keypoints: List[str]=None, skeleton: List[List[str]]=None
    ) -> (pd.DataFrame, pd.DataFrame):
        imgdir = correct_dirpath(imgdir)
        list_df = []
        if classes is None:
            classes   = MetadataCatalog.get(self.dataset_name).thing_classes
        if keypoints is None:
            keypoints = MetadataCatalog.get(self.dataset_name).get("keypoint_names") # Noneの場合はNoneとなる
        if skeleton is None:
            skeleton  = MetadataCatalog.get(self.dataset_name).get("keypoint_flip_map") if keypoints is not None else None
        for y in [imgdir + x for x in coco_gt.df_json["images_file_name"].unique()]:
            print(f"eval image: {y}")
            df = self.eval_a_image(y, coco_gt, classes, thre_iou=thre_iou, keypoints=keypoints, skeleton=skeleton)
            list_df.append(df)
        df_org = pd.concat(list_df, axis=0, ignore_index=True, sort=False)

        # df_org の解析
        list_df = []
        for thre in np.arange(0, 1, 0.1):
            list_df.append(self.eval_with_custom_dataframe(df_org, coco_gt, thre, classes=classes, keypoints=keypoints, skeleton=skeleton))
        df_ana = pd.concat(list_df, axis=0, sort=False, ignore_index=True)
        df_ana = df_ana[list_df[0].columns]
        return df_ana, df_org


    def eval_with_custom_dataframe(
        self, df_org: pd.DataFrame, coco_gt: CocoManager, thre: float, 
        classes: List[str]=None, keypoints: List[str]=None, skeleton: List[List[str]]=None
    ) -> pd.DataFrame:
        if classes is None:
            classes   = MetadataCatalog.get(self.dataset_name).thing_classes
        if keypoints is None:
            keypoints = MetadataCatalog.get(self.dataset_name).get("keypoint_names") # Noneの場合はNoneとなる
        df_ana = pd.DataFrame()
        df_gt  = pd.DataFrame(index=coco_gt.df_json["images_file_name"].unique())
        for class_name in classes:
            df_gt_cls = df_gt.copy()
            df_gt_cls["class"] = class_name
            df_gt_cls["gt"]    = coco_gt.df_json[coco_gt.df_json["categories_name"]==class_name].groupby("images_file_name").size()
            se = pd.Series(dtype=object)
            se["thre"]  = thre
            se["class"] = class_name
            df = df_org[(df_org["score"] >= thre) & (df_org["class"] == class_name)].copy()
            # bbox tp
            labels = df["pred_"+class_name].unique()
            df["pred_maxiou"] = np.nan
            for label in labels:
                if label < 0 or np.isnan(label): continue
                ## max iou のみ tp をつける
                dfwk = df[df["pred_"+class_name] == label].copy()
                dfwk = dfwk.sort_values(by=["gt_iou_"+class_name+"_"+str(int(label))], ascending=False)
                index_max_iou = dfwk.groupby("file_name").apply(lambda x: x.index[0]).values
                dfwk["pred_maxiou"] = "fp"
                dfwk.loc[index_max_iou, "pred_maxiou"] = "tp"
                df.loc[dfwk.index, "pred_maxiou"] = dfwk["pred_maxiou"].copy()
            df["pred_maxiou"] = df["pred_maxiou"].fillna("fp")
            df_piv = df.pivot_table(index="file_name", columns="pred_maxiou", values="score", aggfunc="count").fillna(0)
            if df_piv.columns.isin(["tp"]).sum() == 0: df_piv["tp"] = 0 # ない場合もある
            if df_piv.columns.isin(["fp"]).sum() == 0: df_piv["fp"] = 0 # ない場合もある
            df_piv["gt_n"]      = df.groupby("file_name")["gt_"+class_name+"_n"].first()
            df_piv["pred_n"]    = df_piv["tp"] + df_piv["fp"]
            df_piv["recall"]    = df_piv["tp"] / df_piv["gt_n"]
            df_piv["precision"] = df_piv["tp"] / df_piv["pred_n"]
            # keypoint
            if (df.columns == "pred_keys_"+class_name).sum() > 0:
                for i_key, key_name in enumerate(keypoints):
                    df["key_diff_"+key_name] = df["pred_keys_"+class_name].map(lambda x: x[i_key*2] if type(x) == list else np.nan)
                    sewk = df[(df["pred_maxiou"] == "tp") & (df["key_diff_"+key_name] > 0)].groupby("file_name")["key_diff_"+key_name].mean()
                    df_piv["key_diff_"+key_name] = sewk.copy()
            if (df.columns == "pred_sklt_"+class_name).sum() > 0:
                for i_key, [_, _] in enumerate(skeleton):
                    df["key_sklt_"+str(i_key)] = df["pred_sklt_"+class_name].map(lambda x: x[i_key][-1] if type(x) == list else np.nan)
                    sewk = df[(df["pred_maxiou"] == "tp")].groupby("file_name")["key_sklt_"+str(i_key)].mean()
                    df_piv["key_sklt_ratio_"+str(i_key)] = sewk.copy()
            for _x in df_piv.columns:
                df_gt_cls[_x] = df_piv[_x].copy()
            # summary
            se["gt"]        = df_gt_cls["gt"].fillna(0).sum()
            se["tp"]        = df_gt_cls["tp"].fillna(0).sum()
            se["fp"]        = df_gt_cls["fp"].fillna(0).sum()
            se["recall"]    = df_gt_cls["recall"].fillna(0).mean()
            se["precision"] = df_gt_cls["precision"].mean() # precision は fiillna しちゃだめ
            for keycol in df_gt_cls.columns[df_gt_cls.columns.str.contains("key_diff_", regex=True)]:
                se[keycol]  = df_gt_cls[keycol].mean()
            for keycol in df_gt_cls.columns[df_gt_cls.columns.str.contains("key_sklt_ratio_", regex=True)]:
                se[keycol]  = df_gt_cls[keycol].mean()
            df_ana = df_ana.append(se, ignore_index=True)
        df_ana["f1"] = 2 * df_ana["precision"] * df_ana["recall"] / (df_ana["precision"] + df_ana["recall"])
        colnames_kpt1 = df_ana.columns[df_ana.columns.str.contains("^key_diff_", regex=True)].tolist()
        colnames_kpt2 = df_ana.columns[df_ana.columns.str.contains("^key_sklt_ratio_", regex=True)].tolist()
        df_ana = df_ana[["class","thre","gt","tp","fp","precision","recall","f1"] + colnames_kpt1 + colnames_kpt2]
        return df_ana


    def evalation(self, img_paths: List[str]):
        df = pd.DataFrame()
        for x in img_paths:
            print(x)
            img = cv2.imread(x)
            dfwk = self.predict_to_df(img)
            if dfwk.shape[0] > 0:
                dfwk["image_path"] = x
            else:
                dfwk = pd.DataFrame([x], columns=["image_path"])
            df = pd.concat([df, dfwk], ignore_index=True, sort=False)
        return df


def create_model(args: dict):
    """
    Usage::
        train.py
            from kkutils.util.com import get_args
            args = get_args()
            model(args)
        python train.py  --cocot ./train/coco.json --train ./train/ --outdir ./output --save 200 \
        --dname mark ----classes bracket distribution_board distributor downlight interphone light outlet plumbing spotlight wiring \
        --iter 5000 --batch 5 --njob 5 --model_zoo Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml --aug ./config.json \
        ----minsize 402 434 468 500 532 564 596 --maxsize 1033 --lr 0.01 --warmup 200 \
        ---bbox --cocov ./test_crop/coco.json --valid ./test_crop/ --batchv 4
    """
    # load coco file
    coco = CocoManager()
    if args.get("cocot") is not None:
        coco.add_json(args.get("cocot"))
    outdir       = args.get("outdir") if args.get("outdir") is not None else "./output/"
    imgdir_train = args.get("train")
    imgdir_valid = args.get("valid")
    weight_path  = args.get("weight") # 設定がなかったらNoneになる
    model_zoo    = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml" if args.get("model_zoo") is None else args.get("model_zoo")
    classes      = args.get("classes") if args.get("classes") is not None else [args.get("dname")]
    input_size   = None
    if args.get("train") is not None:
        input_size = (
            tuple([int(x) for x in args.get("minsize")]) if args.get("minsize") is not None and isinstance(args.get("minsize"), list) else (int(args.get("minsize")) if args.get("minsize") is not None and isinstance(args.get("minsize"), str) else (640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960)), 
            tuple([int(x) for x in args.get("maxsize")]) if args.get("maxsize") is not None and isinstance(args.get("maxsize"), list) else (int(args.get("maxsize")) if args.get("maxsize") is not None and isinstance(args.get("maxsize"), str) else 1333)
        )
    else:
        input_size = (
            tuple([int(x) for x in args.get("minsize")]) if args.get("minsize") is not None and isinstance(args.get("minsize"), list) else (int(args.get("minsize")) if args.get("minsize") is not None and isinstance(args.get("minsize"), str) else 800), 
            tuple([int(x) for x in args.get("maxsize")]) if args.get("maxsize") is not None and isinstance(args.get("maxsize"), list) else (int(args.get("maxsize")) if args.get("maxsize") is not None and isinstance(args.get("maxsize"), str) else 1333)
        )
    det2 = MyDet2(
        dataset_name=args.get("dname") if args.get("dname") is not None else "mydataset",
        coco_json_path=args.get("cocot"),
        image_root=imgdir_train,
        outdir=outdir,
        model_zoo_path=model_zoo,
        input_size=input_size, 
        weight_path=weight_path,
        resume=False if args.get("resume") is None else True,
        classes=classes,
        threshold=0.8 if args.get("thre") is None else float(args.get("thre")), 
        max_iter=1000 if args.get("iter") is None else int(args.get("iter")),
        is_train=True if args.get("train") is not None else False, 
        is_bbox_only=False if args.get("bbox") is None else True,
        save_step=5000 if args.get("save") is None else int(args.get("save")),
        aug_json_file_path=None if args.get("aug") is None else args.get("aug"), 
        base_lr=0.001 if args.get("lr") is None else float(args.get("lr")),
        lr_steps=None if args.get("step") is None else (int(args.get("step")[0]), int(args.get("step")[1]), ),
        lr_warmup=1000 if args.get("warmup") is None else int(args.get("warmup")),
        batch_size=2 if args.get("batch") is None else int(args.get("batch")), 
        num_workers=2 if args.get("njob") is None else int(args.get("njob")),
        validations=[("valid1", args.get("cocov"), imgdir_valid), ] if args.get("cocov") is not None else None, 
        valid_steps=100, valid_ndata=int(args.get("batchv")) if args.get("batchv") is not None else 1,
    )
    return det2



class Validator(HookBase):
    def __init__(self, cfg: CfgNode, dataset_name: str, trainer: DefaultTrainer, steps: int=10, ndata: int=5):
        super().__init__()
        self.cfg = cfg.clone()
        self.cfg.DATASETS.TRAIN = (dataset_name, )
        self._loader = iter(build_detection_train_loader(self.cfg))
        self.trainer = trainer
        self.steps = steps
        self.ndata = ndata
        self.loss_dict = {}
        self.data_time = 0
        
    def before_step(self):
        # before に入れないと、after step の後の storage.step で storage._latest~~ が初期化されてしまう
        if self.loss_dict:
            self.trainer._trainer._write_metrics(self.loss_dict, self.data_time)

    def after_step(self):
        if self.trainer.iter > 0 and self.trainer.iter % self.steps == 0:
            list_dict = []
            # self.trainer.model.eval() # これをすると model(data) の動作が変わるのでやらない。
            start = time.perf_counter()
            with torch.no_grad():
                for _ in range(self.ndata):
                    data = next(self._loader)
                    loss_dict = self.trainer.model(data)
                    list_dict.append({
                        k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
                        for k, v in loss_dict.items()
                    })
            loss_dict = {}
            for key in list_dict[0].keys():
                loss_dict[key] = np.mean([dictwk[key] for dictwk in list_dict])
            loss_dict = {
                self.cfg.DATASETS.TRAIN[0] + "_" + k: torch.tensor(v.item()) for k, v in comm.reduce_dict(loss_dict).items()
            }
            self.loss_dict = loss_dict
            self.data_time = time.perf_counter() - start



from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer
class Det2Debug(DefaultTrainer):
    """
    https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=ZyAvNCJMmvFF
    公式の tutorial を参考にして作成した最も simple な class. debug で使用する
    """
    def __init__(self, dataset_name: str=None, coco_json_path: str=None, image_root: str=None, is_predictor: bool=False):
        self._dataset_name = dataset_name
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
        if is_predictor:
            self.cfg = cfg
            self.set_predictor(is_original=True)
        else:
            if self._dataset_name is None: raise Exception("dataset name is None !")
            cfg.DATASETS.TRAIN = (dataset_name,)
            cfg.DATASETS.TEST = ()
            cfg.DATALOADER.NUM_WORKERS = 2
            cfg.SOLVER.IMS_PER_BATCH = 2
            cfg.SOLVER.BASE_LR = 0.01  # pick a good LR
            cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
            cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10  # only has one class (ballon)
            register_coco_instances(dataset_name, {}, coco_json_path, image_root)
            super().__init__(self.cfg)
            self.resume_or_load(resume=False)
    
    def set_predictor(self, is_original: bool=False):
        if is_original == False:
            self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, "model_final.pth")
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
        if self._dataset_name is not None:
            self.cfg.DATASETS.TEST = (self._dataset_name, )
        self.predictor = DefaultPredictor(self.cfg)        

    def show(self, file_path: str):
        metadata = MetadataCatalog.get(self._dataset_name)
        im = cv2.imread(file_path)
        outputs = self.predictor(im)
        v = Visualizer(
            im[:, :, ::-1],
            metadata=metadata, 
            scale=1.0, 
            instance_mode=None   # remove the colors of unsegmented pixels
        )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow("test", v.get_image()[:, :, ::-1])
        cv2.waitKey(0)

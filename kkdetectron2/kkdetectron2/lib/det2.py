import os, datetime
import numpy as np
import cv2
from typing import List

# detectron2
import detectron2
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.data import build_detection_train_loader, DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.logger import setup_logger
setup_logger()

# local package
from kkimagemods.util.common import makedirs, correct_dirpath
from kkimagemods.lib.coco import coco_info, CocoManager
from kkimagemods.util.images import drow_bboxes

class MyDet2(DefaultTrainer):
    def __init__(
            self,
            # coco dataset
            train_dataset_name: str=None, test_dataset_name: str=None, coco_json_path: str=None, image_root: str=None,
            # train params
            cfg=None, mapper=None, max_iter: int=100, is_train: bool=True,
            # train and test params
            model_zoo_path: str="COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml", weight_path: str=None, 
            n_classes: int=1, input_size: tuple=(800, 1333), threshold: float=0.2, outdir: str="./output"
        ):
        # coco dataset
        self.train_dataset_name = train_dataset_name
        self.coco_json_path     = coco_json_path
        self.coco_json_path_org = coco_json_path
        self.image_root         = image_root
        self.mapper             = mapper

        if is_train:
            # train setting
            ## Coco dataset setting
            self.__register_coco_instances()
            self.cfg = cfg if cfg is not None else self.set_config(model_zoo_path, train_dataset_name, weight_path=weight_path, test_dataset_name=test_dataset_name, threshold=threshold, max_iter=max_iter, n_classes=n_classes, input_size=input_size, outdir=outdir)
            os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True) # この宣言は先にする
            super().__init__(self.cfg)
            self.__set_dataloader()
            self.predictor = None
            self.resume_or_load(resume=False) # Falseだとload the model specified by the config (skip all checkpointables).
        else:
            # test setting
            self.cfg = cfg if cfg is not None else self.set_config_basic(model_zoo_path, n_classes, input_size, outdir=outdir)
            self.cfg.DATASETS.TEST = (test_dataset_name, )
            self.set_predictor(weight_path, threshold=threshold)


    def __register_coco_instances(self):
         # この関数で内部のDatasetCatalog, MetadataCatalogにCoco情報をset している
        register_coco_instances(self.train_dataset_name, {}, self.coco_json_path, self.image_root)
    

    def __set_dataloader(self):
        if self.mapper is not None:
            self.data_loader       = build_detection_train_loader(self.cfg, mapper=self.mapper)
            self._data_loader_iter = iter(self.data_loader)


    @classmethod
    def set_config_basic(cls, model_zoo_path, n_classes, input_size: tuple, outdir: str="./output"):
        """
        see https://detectron2.readthedocs.io/modules/config.html#detectron2.config.CfgNode
        """
        ## predict するのに最低限これだけの記述が必要
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(model_zoo_path))
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = n_classes # テスト時はなんか指定しないと動かなかった？
        cfg.OUTPUT_DIR = outdir
        cfg.INPUT.MIN_SIZE_TEST = input_size[0]
        cfg.INPUT.MAX_SIZE_TEST = input_size[1]
        return cfg


    @classmethod
    def set_config(cls, model_zoo_path: str, train_dataset_name: str, weight_path: str=None, test_dataset_name: str=None, threshold: float=0.2, max_iter: int=100, n_classes: int=1, input_size: tuple=(800,1333,), outdir: str="./output"):
        if model_zoo_path is None or train_dataset_name is None: raise Exception("train_dataset_name is needed !!")
        cfg = cls.set_config_basic(model_zoo_path, n_classes, input_size, outdir=outdir)
        #cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        #cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
        cfg.DATASETS.TRAIN = (train_dataset_name, ) # DatasetCatalog, MetadataCatalog の中で自分でsetした"my_dataset_train"を指定
        cfg.DATASETS.TEST  = ((test_dataset_name if test_dataset_name is not None else train_dataset_name),)
        cfg.MODEL.WEIGHTS = (model_zoo.get_checkpoint_url(model_zoo_path)) if weight_path is None else weight_path
        #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  #  Let training initialize from model zoo
        #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
        cfg.SOLVER.BASE_LR = 0.001 # pick a good LR
        cfg.INPUT.MIN_SIZE_TRAIN = input_size[0]
        cfg.INPUT.MAX_SIZE_TRAIN = input_size[1]
        cfg.DATALOADER.NUM_WORKERS = 1
        cfg.SOLVER.IMS_PER_BATCH   = 1
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 # faster, and good enough for this toy dataset (default: 512)
        cfg.SOLVER.MAX_ITER = max_iter    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold   # set the testing threshold for this model
        #cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(MetadataCatalog.get(train_dataset_name).thing_classes)

        return cfg


    def get_predictor(self) -> DefaultPredictor:
        self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, "model_final.pth")
        predictor = DefaultPredictor(self.cfg)
        return predictor
    

    def set_predictor(self, weight_path, threshold: float=None):
        self.cfg.OUTPUT_DIR    = os.path.dirname(weight_path)
        self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, os.path.basename(weight_path))
        if threshold is not None:
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold   # set the testing threshold for this model
        self.predictor = DefaultPredictor(self.cfg)


    def train(self):
        makedirs(self.cfg.OUTPUT_DIR, exist_ok=True, remake=True)
        super().train()
        self.predictor = self.get_predictor()
    

    def predict(self, data: np.ndarray):
        if self.predictor is None:
            self.predictor = self.get_predictor()
        return self.predictor(data)
    

    def predict_and_bbox_image(self, data: np.ndarray) -> List[np.ndarray]:
        output_list = []
        # 推論
        output = self.predict(data)
        output = output["instances"]
        ndf = output.get("pred_boxes").to("cpu").tensor.numpy().copy()
        for x1, y1, x2, y2 in ndf:
            img = data[int(y1):int(y2), int(x1):int(x2), :].copy()
            output_list.append(img)
        return output_list


    def show(self, img: np.ndarray) -> np.ndarray:
        from detectron2.data import MetadataCatalog
        metadata = MetadataCatalog.get(self.cfg.DATASETS.TEST[0])
        output = self.predict(img)
        v = Visualizer(img[:, :, ::-1],
                metadata=metadata, 
                scale=0.8, 
                instance_mode=None #ColorMode.IMAGE_BW # remove the colors of unsegmented pixels
        )
        v = v.draw_instance_predictions(output["instances"].to("cpu"))
        return v.get_image()[:, :, ::-1]


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
    def show_output_dataloader(cls, data):
        """
        dataloader で読み出した画像を解釈するためのクラス
        """
        img = data["image"].detach().numpy().copy().T.astype(np.uint8)
        img = np.rot90(img, 1)
        img = np.flipud(img)
        bbox_list = data["instances"].get("gt_boxes").tensor.numpy().copy()
        return drow_bboxes(img, bbox_list.tolist(), bbox_type="xy")


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
        del DatasetCatalog. _REGISTERED[  self.train_dataset_name] # key を削除しないと再登録できない
        del MetadataCatalog._NAME_TO_META[self.train_dataset_name] # key を削除しないと再登録できない
        self.__register_coco_instances()
        self.__set_dataloader()
        makedirs(outdir, exist_ok=True, remake=True)
        count = 0
        for i, x in enumerate(self.data_loader):
            for j, data in enumerate(x):
                # x には per batch 分の size (2個とか) 入っているので、それ分回す
                img = self.show_output_dataloader(data)
                cv2.imwrite(outdir + "preview_augmentation." + str(i) + "." + str(j) + ".png", img)
            count += 1
            if count > n_output: break

        del DatasetCatalog. _REGISTERED[  self.train_dataset_name] # key を削除しないと再登録できない
        del MetadataCatalog._NAME_TO_META[self.train_dataset_name] # key を削除しないと再登録できない
        self.coco_json_path = self.coco_json_path_org
        self.__register_coco_instances()
        self.__set_dataloader()

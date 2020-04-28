import os, datetime
import numpy as np
import cv2
from typing import List

# detectron2
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.data import build_detection_train_loader
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.logger import setup_logger
setup_logger()

# my modules
from myimagemods.util.common import makedirs
from myimagemods.lib.coco import coco_info, CocoManager


class MyDet2(DefaultTrainer):
    def __init__(self, train_dataset_name="dummy", test_dataset_name=None, cfg=None, mapper=None, weight_path=None, threshold=0.2, max_iter=100):
        self.cfg = cfg if cfg is not None else self.set_config(train_dataset_name, test_dataset_name=test_dataset_name, max_iter=max_iter)
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True) # この宣言は先にする
        if weight_path is None:
            if train_dataset_name == "": raise Exception("train_dataset_name is needed.")
            super().__init__(self.cfg)
            if mapper is not None:
                self.data_loader = build_detection_train_loader(self.cfg, mapper=mapper)
                self._data_loader_iter = iter(self.data_loader)
            self.predictor = None
            self.resume_or_load(resume=False) # Falseだとload the model specified by the config (skip all checkpointables).
        else:
            self.set_predictor(weight_path)
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    

    @classmethod
    def set_config(cls, train_dataset_name, test_dataset_name=None, max_iter=100):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        #cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
        cfg.DATASETS.TRAIN = (train_dataset_name, ) # DatasetCatalog, MetadataCatalog の中で自分でsetした"my_dataset_train"を指定
        cfg.DATASETS.TEST  = ((test_dataset_name if test_dataset_name is not None else train_dataset_name),)
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
        #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
        cfg.SOLVER.IMS_PER_BATCH = 2
        cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
        cfg.SOLVER.MAX_ITER = max_iter    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.OUTPUT_DIR = "./output"
        return cfg


    def get_predictor(self) -> DefaultPredictor:
        self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, "model_final.pth")
        predictor = DefaultPredictor(self.cfg)
        return predictor
    

    def set_predictor(self, weight_path, threshold: float=None):
        self.cfg.OUTPUT_DIR = os.path.dirname(weight_path)
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


    def show(self, img: np.ndarray) -> np.ndarray:
        from detectron2.data import MetadataCatalog
        metadata = MetadataCatalog.get(self.cfg.DATASETS.TEST[0])
        output = self.predict(img)
        v = Visualizer(img[:, :, ::-1],
                metadata=metadata, 
                scale=0.8, 
                instance_mode=ColorMode.IMAGE_BW # remove the colors of unsegmented pixels
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
        ndf     = output.get("pred_masks").  to("cpu").detach().numpy().copy()
        ndf_cat = output.get("pred_classes").to("cpu").detach().numpy().copy()
        ndf_sco = output.get("scores")      .to("cpu").detach().numpy().copy()
        for i_index in np.arange(ndf.shape[0]):
            if ndf_sco[i_index] < self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST: continue # threshold より低いscoreは見ない
            dictwk = {}
            ## segmentation
            contours = cv2.findContours(ndf[i_index, ::].astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]
            segmentation = [contour.reshape(-1) for contour in contours]
            dictwk["segmentation"]  = [_x.tolist() for _x in segmentation]
            dictwk["area"]          = sum([cv2.contourArea(_x.reshape(-1, 1, 2)) for _x in segmentation])
            ## bounding box
            x,y,w,h = cv2.boundingRect(ndf[i_index, ::].astype(np.uint8))
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

import os, datetime, copy
import numpy as np
import cv2
from typing import List
import torch

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
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from fvcore.common.config import CfgNode
setup_logger()

from fvcore.common.file_io import PathManager
from PIL import Image

# local package
from kkimagemods.util.common import makedirs, correct_dirpath
from kkimagemods.lib.coco import coco_info, CocoManager
from kkimagemods.util.images import drow_bboxes
from imageaug import AugHandler, Augmenter as aug


class MyDet2(DefaultTrainer):
    def __init__(
            self,
            # coco dataset
            dataset_name: str = None, coco_json_path: str=None, image_root: str=None,
            # train params
            cfg=None, max_iter: int=100, is_train: bool=True, aug_json_file_path: str=None, 
            base_lr: float=0.01, num_workers: int=2, resume: bool=False, 
            # train and test params
            model_zoo_path: str="COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml", weight_path: str=None, 
            classes: List[str] = None, input_size: tuple=(800, 1333), threshold: float=0.2, outdir: str="./output"
        ):
        # coco dataset
        self.dataset_name       = dataset_name
        self.model_zoo_path     = model_zoo_path
        self.coco_json_path     = coco_json_path
        self.coco_json_path_org = coco_json_path
        self.image_root         = image_root
        self.is_train           = is_train
        self.__register_coco_instances() # Coco dataset setting
        self.cfg                = cfg if cfg is not None else self.set_config(
            weight_path=weight_path, threshold=threshold, max_iter=max_iter, num_workers=num_workers, 
            base_lr=base_lr, classes=classes, input_size=input_size, outdir=outdir
        )
        # classes は強制でセットする
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)
        MetadataCatalog.get(self.dataset_name).thing_classes = classes
        self.mapper = None if aug_json_file_path is None else MyMapper(self.cfg, aug_json_file_path, is_train=self.is_train)

        if self.is_train:
            # train setting
            os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True) # この宣言は先にする
            super().__init__(self.cfg) # train の時しか init しない
            self.predictor = None
            self.resume_or_load(resume=resume) # Falseだとload the model specified by the config (skip all checkpointables).

        # この定義は最後がいいかも
        if self.is_train == False:
            # test setting
            self.set_predictor()


    def __register_coco_instances(self):
         # この関数で内部のDatasetCatalog, MetadataCatalogにCoco情報をset している
        register_coco_instances(self.dataset_name, {}, self.coco_json_path, self.image_root)


    # override. super().__init__ 内でこの関数が呼ばれる
    def build_train_loader(self, cfg) -> torch.utils.data.DataLoader:
        return build_detection_train_loader(cfg, mapper=self.mapper)


    def set_config(
        self, weight_path: str=None, threshold: float=0.2, max_iter: int=100, num_workers: int=2, 
        classes: List[str]=None, base_lr: float=0.01, input_size: tuple=(800,1333,), outdir: str="./output"
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
        cfg.SOLVER.IMS_PER_BATCH   = num_workers
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 # faster, and good enough for this toy dataset (default: 512)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold if threshold is not None else 0.2  # set the testing threshold for this model
        cfg.INPUT.MIN_SIZE_TRAIN = input_size[0]
        cfg.INPUT.MAX_SIZE_TRAIN = input_size[1]
        cfg.INPUT.MIN_SIZE_TEST = input_size[0]
        cfg.INPUT.MAX_SIZE_TEST = input_size[1]
        if self.is_train:
            cfg.SOLVER.BASE_LR = base_lr # pick a good LR
            cfg.SOLVER.MAX_ITER = max_iter    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
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


    def img_crop_bbox(self, img: np.ndarray, padding: int=0) -> List[np.ndarray]:
        output_list = []
        # 推論
        output = self.predictor(img)
        output = output["instances"]
        ndf = output.get("pred_boxes").to("cpu").tensor.numpy().copy()
        for x1, y1, x2, y2 in ndf:
            imgwk = img[int(y1)-padding:int(y2)+padding, int(x1)-padding:int(x2)+padding, :].copy()
            output_list.append(imgwk)
        return output_list


    def show(self, img: np.ndarray, add_padding: int=0, preview: bool=False) -> np.ndarray:
        from detectron2.data import MetadataCatalog
        metadata = MetadataCatalog.get(self.dataset_name)
        output   = self.predict(img)
        for i in range(output["instances"].get("pred_boxes").tensor.shape[0]):
            for j in range(4):
                output["instances"].get("pred_boxes").tensor[i][j] += add_padding
        # padding. annotation が見えなくなる場合もあるため
        if add_padding > 0:
            img = cv2.copyMakeBorder(
                img, add_padding, add_padding, add_padding, add_padding,
                cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
        v = Visualizer(img[:, :, ::-1],
            metadata=metadata, 
            scale=0.8, 
            instance_mode=None #ColorMode.IMAGE_BW # remove the colors of unsegmented pixels
        )
        v = v.draw_instance_predictions(output["instances"].to("cpu"))
        img_ret = v.get_image()[:, :, ::-1]
        if preview:
            cv2.imshow(__name__, img_ret)
            cv2.waitKey(0)
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
        del DatasetCatalog. _REGISTERED[  self.dataset_name] # key を削除しないと再登録できない
        del MetadataCatalog._NAME_TO_META[self.dataset_name] # key を削除しないと再登録できない
        self.__register_coco_instances()
        super().__init__(self.cfg)
        makedirs(outdir, exist_ok=True, remake=True)
        count = 0
        for i, x in enumerate(self.data_loader):
            for j, data in enumerate(x):
                # x には per batch 分の size (2個とか) 入っているので、それ分回す
                img = self.show_output_dataloader(data)
                cv2.imwrite(outdir + "preview_augmentation." + str(i) + "." + str(j) + ".png", img)
            count += 1
            if count > n_output: break

        del DatasetCatalog. _REGISTERED[  self.dataset_name] # key を削除しないと再登録できない
        del MetadataCatalog._NAME_TO_META[self.dataset_name] # key を削除しないと再登録できない
        self.coco_json_path = self.coco_json_path_org
        self.__register_coco_instances()
        super().__init__(self.cfg)


class MyMapper(DatasetMapper):
    def __init__(self, cfg, json_file_path, is_train=True):
        super().__init__(cfg, is_train=is_train)
        self.aug_handler = AugHandler.load_from_path(json_file_path)
        if is_train: self.tfm_gens = self.tfm_gens[:-1]
    
    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        ### my code ##
        image, dataset_dict = self.aug_handler(image=image, dataset_dict_detectron=dataset_dict)
        ### my code ##

        if "annotations" not in dataset_dict:
            image, transforms = T.apply_transform_gens(
                ([self.crop_gen] if self.crop_gen else []) + self.tfm_gens, image
            )
        else:
            # Crop around an instance if there are instances in the image.
            # USER: Remove if you don't use cropping
            if self.crop_gen:
                crop_tfm = utils.gen_crop_transform_with_instance(
                    self.crop_gen.get_crop_size(image.shape[:2]),
                    image.shape[:2],
                    np.random.choice(dataset_dict["annotations"]),
                )
                image = crop_tfm.apply_image(image)
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            if self.crop_gen:
                transforms = crop_tfm + transforms

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        # USER: Remove if you don't use pre-computed proposals.
        if self.load_proposals:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, self.min_box_side_len, self.proposal_topk
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                if not self.keypoint_on:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.mask_format
            )
            # Create a tight bounding box from masks, useful when image is cropped
            if self.crop_gen and instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            with PathManager.open(dataset_dict.pop("sem_seg_file_name"), "rb") as f:
                sem_seg_gt = Image.open(f)
                sem_seg_gt = np.asarray(sem_seg_gt, dtype="uint8")
            sem_seg_gt = transforms.apply_segmentation(sem_seg_gt)
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
            dataset_dict["sem_seg"] = sem_seg_gt
        return dataset_dict



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
    def __init__(self, dataset_name: str, coco_json_path: str, image_root: str):
        self._dataset_name = dataset_name
        register_coco_instances(dataset_name, {}, coco_json_path, image_root)
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
        cfg.DATASETS.TRAIN = (dataset_name,)
        cfg.DATASETS.TEST = ()
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")  # Let training initialize from model zoo
        cfg.SOLVER.IMS_PER_BATCH = 2
        cfg.SOLVER.BASE_LR = 0.01  # pick a good LR
        cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10  # only has one class (ballon)
        super().__init__(cfg)
        self.resume_or_load(resume=False)
    
    def set_predictor(self):
        self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, "model_final.pth")
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
        self.cfg.DATASETS.TEST = (self._dataset_name, )
        self.predictor = DefaultPredictor(self.cfg)        

    def show(self, file_path: str):
        metadata = MetadataCatalog.get(self._dataset_name)
        im = cv2.imread(file_path)
        outputs = self.predictor(im)
        v = Visualizer(im[:, :, ::-1],
                    metadata=metadata, 
                    scale=0.8, 
                    instance_mode=None   # remove the colors of unsegmented pixels
        )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow("test", v.get_image()[:, :, ::-1])
        cv2.waitKey(0)
import kkimagemods.util.logger

# import some common libraries
import cv2
import os
from functools import partial

# import local packcage
from kkimagemods.util.common import get_file_list, makedirs, get_args, correct_dirpath
from kkimagemods.lib.coco import CocoManager
from kkdetectron2.lib.det2 import MyDet2, Det2Debug


if __name__ == "__main__":
    coco_json = "../20200804_hook_seg_realdata/coco.json"

    # segmentation は消しておく
    coco = CocoManager()
    coco.add_json(coco_json)
    coco.rmkeypoints(
        ['kpt_a', 'kpt_cb', 'kpt_c', 'kpt_cd', 'kpt_e', 'kpt_b', 'kpt_d'],
        [['kpt_a', 'kpt_cb'], ['kpt_cb', 'kpt_c'], ['kpt_c', 'kpt_cd'], ['kpt_cd','kpt_e'], ['kpt_b', 'kpt_d']],
    )
    coco.organize_segmentation()
    coco.re_index()
    coco.save(  coco_json + ".segkey")
    coco_json = coco_json + ".segkey"
    coco_json_train = coco_json

     # validation data の作成
    coco_json_valid = "../20200727_hook_annotation/level_01.json"
    coco = CocoManager()
    coco.add_json(coco_json_valid)
    coco.rmkeypoints(
        ['kpt_a', 'kpt_cb', 'kpt_c', 'kpt_cd', 'kpt_e', 'kpt_b', 'kpt_d'],
        [['kpt_a', 'kpt_cb'], ['kpt_cb', 'kpt_c'], ['kpt_c', 'kpt_cd'], ['kpt_cd','kpt_e'], ['kpt_b', 'kpt_d']],
    )
    coco.df_json["annotations_segmentation"]  = coco.df_json["annotations_segmentation"].apply(lambda x: [[1,1,2,1,2,2]])
    coco.df_json = coco.df_json[coco.df_json["images_height"] <= coco.df_json["images_width"]].copy() ## なんか height より width の方が大きくないとエラーが発生する
    coco.re_index()
    coco.save(coco_json_valid + ".rmsegkey")
    coco_json_valid = coco_json_valid + ".rmsegkey"

    outdir = "./output/"
    imgdir_train = "../20200804_hook_seg_realdata/traindata/"
    imgdir_valid = "../20200727_hook_annotation/level_01/"
    weight_path  = "./output_20200813_iter35000/model_0029999.pth"
    det2 = MyDet2(
        dataset_name="hook",
        coco_json_path=coco_json_train,
        image_root=imgdir_train,
        outdir=outdir,
        model_zoo_path="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml", #"COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml", 
        input_size=(1024, 1024, ),
        weight_path=weight_path,
        resume=False,
        is_keyseg=True,
        classes=['hook',"pole"],
        keypoint_names = ['kpt_a', 'kpt_cb', 'kpt_c', 'kpt_cd', 'kpt_e', 'kpt_b', 'kpt_d'], 
        keypoint_flip_map = [['kpt_a', 'kpt_cb'], ['kpt_cb', 'kpt_c'], ['kpt_c', 'kpt_cd'], ['kpt_cd','kpt_e'], ['kpt_b', 'kpt_d']],
        threshold=0.01, max_iter=50000, is_train=True, aug_json_file_path="./augmentation.json", 
        base_lr=0.003, num_workers=1, lr_steps=(30000, 40000),
        validations=[ # valid: (dataset_name, json_path, image_path)
            ("valid", coco_json_valid, imgdir_valid),
        ],
        valid_steps=100, valid_ndata=10,
    )

    # augmentation check
    #det2.preview_augmentation("000000.0.png", outdir="./preview_augmentation", n_output=100)
    det2.train()

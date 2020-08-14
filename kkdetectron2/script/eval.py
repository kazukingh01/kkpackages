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
    outdir = "./output_seg/"
    imgdir_train = "../20200804_hook_seg_realdata/traindata/"
    imgdir_test  = "../20200730_hook_key7_test/testdata/"
    weight_path  = "./output/model_0034999.pth"
    det2 = MyDet2(
        dataset_name="hook",
        outdir=outdir,
        model_zoo_path="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml", #"COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml", 
        input_size=(1024, 1024, ),
        weight_path=weight_path,
        resume=False,
        is_keyseg=True,
        classes=['hook',"pole"],
        keypoint_names = ['kpt_a', 'kpt_cb', 'kpt_c', 'kpt_cd', 'kpt_e', 'kpt_b', 'kpt_d'], 
        keypoint_flip_map = [['kpt_a', 'kpt_cb'], ['kpt_cb', 'kpt_c'], ['kpt_c', 'kpt_cd'], ['kpt_cd','kpt_e'], ['kpt_b', 'kpt_d']],
        threshold=0.5, is_train=False, 
    )

    # predict train data
    makedirs(correct_dirpath(outdir) + "train/", exist_ok=True, remake=True)
    for x in get_file_list(imgdir_train, regex_list=[r"jpg$", r"png$", r"JPG$"])[:100]:
        img = cv2.imread(x)
        output = det2.show(img, only_best=False)
        cv2.imwrite(correct_dirpath(outdir) + "train/" + os.path.basename(x), output)

    # predict test data
    makedirs(correct_dirpath(outdir) + "test/", exist_ok=True, remake=True)
    for x in get_file_list(imgdir_test, regex_list=[r"jpg$", r"png$", r"JPG$"]):
        img = cv2.imread(x)
        output = det2.show(img, only_best=False)
        cv2.imwrite(correct_dirpath(outdir) + "test/" + os.path.basename(x), output)

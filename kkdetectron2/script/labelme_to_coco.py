# local package
from kkimagemods.lib.coco import Labelme2Coco


if __name__ == "__main__":
    labelme = Labelme2Coco(
        dirpath_json="./json_labelme_level_01/",
        dirpath_img="./level_01/",
        categories_name=["hook","pole"],
        keypoints=["kpt_a","kpt_cb","kpt_c","kpt_cd","kpt_e","kpt_b","kpt_d","d_left","d_right"],
        keypoints_belong={
            "hook":["kpt_a","kpt_cb","kpt_c","kpt_cd","kpt_e","kpt_b","kpt_d","d_left","d_right"], 
            "pole":[]
        },
        skelton=[["kpt_a","kpt_cb"],["kpt_cb","kpt_c"],["kpt_c","kpt_cd"],["kpt_cd","kpt_e"],["kpt_b","kpt_d"],["d_left","d_right"]]
    )
    labelme.to_coco("./level_01.json")
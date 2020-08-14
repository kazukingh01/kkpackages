from kkimagemods.lib.coco import Ndds2Coco, CocoManager
from kkimagemods.util.common import get_args

if __name__ == "__main__":
    args = get_args()

    inputdir = "./"+args.get("dir")+"/"
    ndds_to_coco = Ndds2Coco(
        indir=inputdir,
        ignore_jsons=["_camera_settings.json","_object_settings.json"],
        setting_json_fname="_object_settings.json"
    )

    ndds_to_coco.set_base_parameter(
        inputdir, n_files=200, 
        instance_merge={
            "hook": ["hook", "kpt_a", "kpt_b", "kpt_cb", "kpt_c", "kpt_cd", "kpt_d", "kpt_e", "hook_mud"],
            "pole": ["pole", "pole_mud"]
        },
        bboxes={
            "hook": "hook_bbox"
        },
        keypoints={
            "hook":{
                "kpt_a" :[{"name":"kpt_a",  "type":"center"}],
                "kpt_b" :[{"name":"kpt_b",  "type":"center"}],
                "kpt_cb":[{"name":"kpt_cb", "type":"center"}],
                "kpt_c" :[{"name":"kpt_c",  "type":"center"}],
                "kpt_cd":[{"name":"kpt_cd", "type":"center"}],
                "kpt_d" :[{"name":"kpt_d",  "type":"center"}],
                "kpt_e" :[{"name":"kpt_e",  "type":"center"}],
            },
        },
        segmentations={
            "hook":{"name": "hook_bbox", "type": "inbox"}
        },
        convert_mode="is"
    )
    ndds_to_coco.read_ndds_output_all(visibility_threshold=-1, max_count=None)
    ndds_to_coco.output_coco_format(
        "./"+args.get("dir")+".json", 
        {
            "supercategory":{
                "hook":["hook"], 
                "pole":["pole"], 
            }, 
            "categories"   :{
                "hook":["hook"],
                "pole":["pole"], 
            },
            "keypoints":["kpt_a","kpt_cb","kpt_c","kpt_cd","kpt_e","kpt_b","kpt_d"],
            "skelton":[[0,1],[1,2],[2,3],[3,4],[5,6]],
        }
    )

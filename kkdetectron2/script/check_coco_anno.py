from kkimagemods.lib.coco import CocoManager
import kkimagemods.util.images
import cv2
import numpy as np
from kkimagemods.util.common import get_args

if __name__ == "__main__":
    """
    Usage::
        python -i ***.py --json ./coco.json --indir ./root_image/
    """
    args = get_args()

    coco_json = args.get("json")
    coco = CocoManager()
    coco.add_json(coco_json)
    coco.output_draw_infomation(
        "./check_annotation/", 
        imgpath=args.get("indir") if args.get("indir") is not None else None,
        exist_ok=True, remake=True, is_anno_name=True
    )

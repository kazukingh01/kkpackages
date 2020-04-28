from kkimagemods.lib.coco import Ndds2Coco

if __name__ == "__main__":
    ndds_to_coco = Ndds2Coco()
    ndds_to_coco.set_base_parameter(
        "./ndds_output", n_files=100, 
        instance_merge={"color_cone1":["colorcone1","colorcone2"], }, 
        visibility_threshold = 0.2
        )
    ndds_to_coco.read_ndds_output_all("./ndds_output")
    ndds_to_coco.sample_ouptut()
    ndds_to_coco.output_coco_format(
        "./output_ndds_to_coco", 
        {"supercategory":{"cone":["color_cone"]}, 
         "categories"   :{"color_cone" :["color_cone1"]}}
    )



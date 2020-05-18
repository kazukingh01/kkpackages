import copy, os
import torch
from detectron2.data import detection_utils as utils
from imageaug import AugHandler, Augmenter as aug

def mapper(dataset_dict, json_file_path = ""):
    # Implement a mapper, similar to the default DatasetMapper, but with your own customizations
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    image = utils.read_image(dataset_dict["file_name"], format="BGR")

    handler = AugHandler.load_from_path(json_file_path)
    image, dataset_dict = handler(image=image, dataset_dict_detectron=dataset_dict)
    
    annots = []

    for item in dataset_dict["annotations"]:
        annots.append(item)

    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
    instances = utils.annotations_to_instances(annots, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)

    return dataset_dict

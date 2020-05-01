import copy, os
import torch
from detectron2.data import detection_utils as utils
from imageaug import AugHandler, Augmenter as aug

def mapper(dataset_dict, json_file_path = ""):
    # Implement a mapper, similar to the default DatasetMapper, but with your own customizations
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    image = utils.read_image(dataset_dict["file_name"], format="BGR")

    # load_augmentation_settings 中で if not file_exists(handler_save_path) となっているので、Defaultは""にする
    handler = load_augmentation_settings(handler_save_path = json_file_path)
    image, dataset_dict = handler(image=image, dataset_dict_detectron=dataset_dict)
    
    annots = []

    for item in dataset_dict["annotations"]:
        annots.append(item)

    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
    instances = utils.annotations_to_instances(annots, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)

    return dataset_dict


def load_augmentation_settings(handler_save_path: str):

    if not os.path.isfile(handler_save_path):
        handler = AugHandler(
            [
                aug.Crop(percent=[0.2, 0.5]),
                aug.Affine(scale = {"x": tuple([0.8, 1.2]), "y":tuple([0.8, 1.2])}, translate_percent= {"x": tuple([0.1, 0.11]), "y":tuple([0.1, 0.11])}, rotate= [-180, 180], order= [0, 0], cval= [0, 0], shear= [0,0])
            ]
        )
        handler.save_to_path(save_path=handler_save_path, overwrite=True)
    else:
        handler = AugHandler.load_from_path(handler_save_path)

    return handler
from torchvision import models
from torch import nn


class MyImageModel(nn.Module):

    def __init__(self):
        self.model_zoo = models.densenet121(pretrained=True) # ImageNetの画像で学習したWeightを使える
        self.add_module("model_zee_nn", self.model_zoo.features)

    
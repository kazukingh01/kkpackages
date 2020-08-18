from torchvision import models
from torch import nn

# local package
from kkreinforce.lib.kknn import TorchNN


class MyImageNet(nn.Module):

    def __init__(self, torch_nn: TorchNN):
        super().__init__()
        nn_trained = models.densenet161(pretrained=True)
        self.add_module("model_zoo", nn_trained.__getattr__("features")) # ImageNetの画像で学習したWeightを使える
        self.add_module("my_nn",     torch_nn)

import torch
from torch import nn
from typing import List
from collections import namedtuple


Layer = namedtuple("Layer", ("name", "module", "node", "calc_type", "params", "kwards"))
class TorchNN(nn.Module):

    def __init__(self, in_size: int, *layers: List[Layer]):
        """
        Layer: ("name", "module", "node", "params", "kwards")
            name: str. module name
            module: class. torch module
            node: int or None. number of output node.
            params: tuple. *params で渡す
            kwards: dict. **kwards で渡す
        """
        super(TorchNN, self).__init__()
        # 計算後の output でどこを使うかの index を持っておく
        self.indexes = [None] # 0 番目は None で埋めておく
        for layer in layers:
            self.indexes.append(layer.calc_type)
            if   layer.node is None:
                self.add_module(layer.name, layer.module(*layer.params, **layer.kwards))
            elif layer.node == 0:
                self.add_module(layer.name, layer.module(in_size, *layer.params, **layer.kwards))
            else:
                self.add_module(layer.name, layer.module(in_size, layer.node, *layer.params, **layer.kwards))
                in_size = layer.node
    

    def __call__(self, input: torch.Tensor, option: str=None):
        output = input.clone()
        for i, module in enumerate(self.modules()):
            if i == 0: continue
            output = module(output)
            if   self.indexes[i] is None:
                pass
            elif self.indexes[i] == "reshape(x,-1)":
                output = output.reshape(output.shape[0], -1)
            elif self.indexes[i] == "rnn_outonly":
                output = output[0]
            elif self.indexes[i] == "rnn_last":
                output = output[:, -1, :]
            elif self.indexes[i] == "rnn_all":
                output = output.reshape(-1, output.shape[-1])
            elif self.indexes[i] == "call_options":
                if   option is not None and option == "not_first":
                    output = output[:, 1:, :]
                elif option is not None and option == "last":
                    output = output[:, -1, :]
            else:
                raise Exception(f'calc type: {self.indexes[i]} is not expected.')
        return output



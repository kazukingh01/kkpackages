from collections import namedtuple
import torch
from torch import nn
from typing import List

# locla package
from kkutils.util.com import set_logger, is_callable
logger = set_logger(__name__)


__all__ = [
    "Layer",
    "TorchNN",
]


Layer = namedtuple("Layer", ("name", "module", "node", "calc_type", "params", "kwards"))
class TorchNN(nn.Module):
    def __init__(self, in_size: int, *layers: List[Layer]):
        """
        Params::
            in_size: number of input nodes.
            *layers: list of "Layer"
                Layer: ("name", "module", "node", "params", "kwards")
                    name: str. module name
                    module: class. torch module
                    node: int or None. number of output node.
                    params: tuple. *params で渡す
                    kwards: dict. **kwards で渡す
        Usage::
        >>> mynn = TorchNN(
                128,
                Layer("", nn.Linear,   64,   None, (), {}),
                Layer("", nn.ReLU,     None, None, (), {}),
                Layer("", nn.Identity, None, "split_2", (), {}), #途中で分かれる. 分かれた場合、その数の先のLayerまでを現在のcopyで計算する
                Layer("", nn.Linear(64, 3), None, None, (), {}), #分岐中はInstanceを作成する
                Layer("", nn.Linear(64, 3), None, None, (), {}), #分岐中はInstanceを作成する
                Layer("", nn.Identity, 6, "combine", (), {}), # また合体
                Layer("", nn.Linear,   2,   None, (), {}),
                Layer("", nn.Softmax,None, None, (), {"dim":1}),
            )
        >>> input_ = torch.rand(100,128)
        >>> mynn(input_).shape
        torch.Size([100, 2])
        """
        super(TorchNN, self).__init__()
        # 計算後の output でどこを使うかの index を持っておく
        self.indexes = []
        self.modnames = [] # moduleのaddressを格納する
        self.list_modules = [] # moduleのaddressを格納する
        self.list_split_output    = []
        """
        例えば下記形式になっている。Tupleの0番目は、何番目のsplitか、1番目はその番目での参照indexが格納されている
        >>> self.index_split_output
        ((None, None), (None, None), (None, None), (0, -3), (0, -2), (0, -1))
        """
        self.index_split_output   = []
        self.index_combine_output = []
        self.proc_main  = []
        self.proc_after = []
        
        for i_layer, layer in enumerate(layers):
            name = TorchNN.__name__ + str(i_layer).zfill(3) if layer.name is None or layer.name == "" else layer.name
            logger.debug(f'Layer: {layer}')
            self.indexes.append(layer.calc_type)
            self.modnames.append(name)
            if   layer.node is None:
                if isinstance(layer.module, nn.Module):
                    self.add_module(name, layer.module)
                else:
                    self.add_module(name, layer.module(*(() if not layer.params else layer.params), **({} if not layer.kwards else layer.kwards)))
            elif layer.node == 0:
                self.add_module(name, layer.module(in_size, *(() if not layer.params else layer.params), **({} if not layer.kwards else layer.kwards)))
            else:
                self.add_module(name, layer.module(in_size, layer.node, *(() if not layer.params else layer.params), **({} if not layer.kwards else layer.kwards)))
                in_size = layer.node
            self.list_modules.append(None)
            self.list_modules[-1] = self.__getattr__(name)
        self.compile()
    
    def compile(self):
        split_cnt = 0
        for i, _ in enumerate(self.list_modules):
            # main proc の compile
            if split_cnt == 0:
                self.index_split_output.append((None,None,))
                self.proc_main.append(lambda module, output: module(output))
            elif split_cnt > 0:
                self.index_split_output.append((len(self.list_split_output) - 1, -split_cnt,))
                self.proc_main.append(lambda module, output: module(output))
                split_cnt += -1 # 1 ずつ減らす.
            else:
                raise Exception(f'split type: {self.indexes[i]} is not expected.')            

            # after proc の compile
            self.index_combine_output.append(None)
            if   not self.indexes[i]:
                self.proc_after.append(lambda x, opt: x)
            elif self.indexes[i] == "reshape(x,-1)":
                self.proc_after.append(lambda x, opt: x.reshape(x.shape[0], -1))
            elif self.indexes[i].find("split_") == 0:
                self.proc_after.append(lambda x, opt: x)
                self.list_split_output.append([])
                split_cnt = int(self.indexes[i][-1])
                for _ in range(split_cnt): self.list_split_output[-1].append(None)
            elif self.indexes[i] == "combine":
                self.index_combine_output[-1] = len(self.list_split_output) - 1
                self.proc_after.append(lambda x, opt: torch.cat([_x.reshape(_x.shape[0], -1) for _x in x], dim=1))
            elif self.indexes[i] == "out_split":
                self.proc_after.append(lambda x, opt: self.list_split_output[-1])
            elif self.indexes[i] == "rnn_outonly":
                self.proc_after.append(lambda x, opt: x[0])
            elif self.indexes[i] == "rnn_last":
                self.proc_after.append(lambda x, opt: x[:, -1, :])
            elif self.indexes[i] == "rnn_all":
                self.proc_after.append(lambda x, opt: x.reshape(-1, x.shape[-1]))
            elif self.indexes[i] == "debug":
                def work(x, opt):
                    logger.info(f'\n{x}, \nshape: {x.shape}')
                    return x
                self.proc_after.append(work)
            elif self.indexes[i] == "call_options":
                self.proc_after.append(lambda x, opt: x if opt is None else (x[:, 1:, :] if opt == "not_first" else (x[:, -1, :] if opt == "last" else None)))
            else:
                raise Exception(f'calc type: {self.indexes[i]} is not expected.')            
        # Tuple で高速化
        self.index_split_output   = tuple(self.index_split_output)
        self.index_combine_output = tuple(self.index_combine_output)
        self.proc_main  = tuple(self.proc_main)
        self.proc_after = tuple(self.proc_after)

    def forward(self, _input: torch.Tensor, option: str=None):
        output = _input.clone()
        for i, module in enumerate(self.list_modules):
            # DEBUG CODE を入れると劇遅になるので注意
            # logger.debug(f'module: {self.modnames[i]}, {module}, \ninput: {type(output)}, {output.shape if is_callable(output, "shape") else len(output)}\n{output}')
            if self.index_split_output[i][0] is None:
                output = self.proc_main[i](module, output)
            else:
                self.list_split_output[self.index_split_output[i][0]][self.index_split_output[i][1]] = self.proc_main[i](module, output)
            if self.index_combine_output[i] is None:
                output = self.proc_after[i](output, option)
            else:
                output = self.list_split_output[self.index_combine_output[i]]
                output = self.proc_after[i](output, option)
        return output

    @classmethod
    def _reset_parameters(cls, module, weight: float=None):
        try:
            module.reset_parameters()
        except AttributeError:
            pass
        if weight is None: pass
        elif isinstance(weight, str) and weight == "norm":
            if is_callable(module, "weight"):
                torch.nn.utils.weight_norm(module, "weight")
            if is_callable(module, "bias"):
                torch.nn.utils.weight_norm(module, "bias")
        else:
            try:
                module.weight.data.fill_(weight)
            except AttributeError:
                pass
            try:
                module.bias.data.fill_(weight)
            except AttributeError:
                pass

    def reset_parameters(self, weight: float=None):
        """
        Params::
            weight: float or str. string で "random" の場合は reset_parameters() を呼び出す
                norm: weght normalization という手法. 
        """
        for name, module in self.named_modules():
            if name != "":
                if isinstance(module, TorchNN): continue
                logger.info(f"reset params: {name}")
                self._reset_parameters(module, weight=weight)


class ResBlock(TorchNN):
    def forward(self, _input: torch.Tensor, option: str=None):
        output = super().forward(_input, option=option)
        output = output + _input
        return output


class Mish(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input * torch.tanh(torch.nn.functional.softplus(input))

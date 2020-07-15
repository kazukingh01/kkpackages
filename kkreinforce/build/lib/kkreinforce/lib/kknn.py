import torch
from torch import nn
from typing import List
from collections import namedtuple
# local package
from kkimagemods.util.common import is_callable
from kkimagemods.util.logger import set_loglevel, set_logger
logger = set_logger(__name__)


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
        self.indexes = []
        self.modnames = [] # moduleのaddressを格納する
        self.list_modules = [] # moduleのaddressを格納する
        for layer in layers:
            logger.debug(f'Layer: {layer}')
            self.indexes.append(layer.calc_type)
            self.modnames.append(layer.name)
            if   layer.node is None:
                if type(layer.module) == TorchNN:
                    self.add_module(layer.name, layer.module)
                else:
                    self.add_module(layer.name, layer.module(*layer.params, **layer.kwards))
            elif layer.node == 0:
                self.add_module(layer.name, layer.module(in_size, *layer.params, **layer.kwards))
            else:
                self.add_module(layer.name, layer.module(in_size, layer.node, *layer.params, **layer.kwards))
                in_size = layer.node
            self.list_modules.append(None)
            self.list_modules[-1] = self.__getattr__(layer.name)

            # Compile
            self.list_split_output    = []
            self.index_split_output   = []
            self.index_combine_output = []
            self.proc_main  = []
            self.proc_after = []
            self.compile()
    

    def compile(self):
        self.list_split_output    = []
        self.index_split_output   = []
        self.index_combine_output = []
        self.proc_main  = []
        self.proc_after = []
        split_tuple_cnt, split_numpy_cnt = 0, 0
        for i, _ in enumerate(self.list_modules):
            # main proc の compile
            if split_tuple_cnt == 0 and split_numpy_cnt == 0:
                split_tuple_cnt, split_numpy_cnt = 0, 0
                self.index_split_output.append((None,None,))
                self.proc_main.append(lambda module, output: module(output))
            elif split_tuple_cnt > 0:
                self.index_split_output.append((len(self.list_split_output) - 1, -split_tuple_cnt,))
                self.proc_main.append(lambda module, output: module(output[-split_tuple_cnt]))
                split_tuple_cnt += -1 # 1 ずつ減らす.
            elif split_numpy_cnt > 0:
                self.index_split_output.append((len(self.list_split_output) - 1, -split_numpy_cnt,))
                self.proc_main.append(lambda module, output: module(output[:, -split_tuple_cnt]))
                split_numpy_cnt += -1 # 1 ずつ減らす.
            else:
                raise Exception(f'split type: {self.indexes[i]} is not expected.')            

            # after proc の compile
            self.index_combine_output.append(None)
            if   self.indexes[i] is None:
                self.proc_after.append(lambda x, opt: x)
            elif self.indexes[i] == "reshape(x,-1)":
                self.proc_after.append(lambda x, opt: x.reshape(x.shape[0], -1))
            elif self.indexes[i].find("split_tuple_") == 0:
                self.proc_after.append(lambda x, opt: x)
                self.list_split_output.append([])
                split_tuple_cnt = int(self.indexes[i][-1])
                for _ in range(split_tuple_cnt): self.list_split_output[-1].append(None)
            elif self.indexes[i].find("split_numpy_") == 0:
                self.proc_after.append(lambda x, opt: x)
                self.list_split_output.append([])
                split_numpy_cnt = int(self.indexes[i][-1])
                for _ in range(split_numpy_cnt): self.list_split_output[-1].append(None)
            elif self.indexes[i] == "combine":
                self.index_combine_output[-1] = len(self.list_split_output) - 1
                self.proc_after.append(lambda x, opt: torch.cat([_x.reshape(_x.shape[0], -1) for _x in x], dim=1))
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
            # DEBUG CODE を入れると劇遅になる
            #logger.debug(f'module: {self.modnames[i]}, {module}, \ninput: {type(output)}, {output.shape if is_callable(output, "shape") else len(output)}\n{output}')
            if self.index_split_output[i][0] is None:
                output = self.proc_main[i](module, output)
            else:
                self.list_split_output[self.index_split_output[i][0]][self.index_split_output[i][1]] = self.proc_main[i](module, output)
            if self.index_combine_output[i] is None:
                output = self.proc_after[i](output, option)
            else:
                output = self.list_split_output[self.index_combine_output[i]]
                output = self.proc_after[i](output, option)
            #logger.debug(f'\noutput: {type(output)}, {output.shape if is_callable(output, "shape") else len(output)}\n{output}')
        return output


    def set_weight(self, weight: float):
        # 重みの初期化
        for name, _ in self.named_modules():
            if name != "":
                try:
                    self.__getattr__(name).weight.data.fill_(weight)
                except AttributeError:
                    pass
                try:
                    self.__getattr__(name).bias.data.fill_(weight)
                except AttributeError:
                    pass

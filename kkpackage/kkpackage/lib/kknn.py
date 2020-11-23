import datetime
from typing import List, Tuple
from collections import namedtuple
from functools import partial
import numpy as np
import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter
# local package
from kkpackage.util.dataframe import divide_index
from kkpackage.util.common import is_callable, correct_dirpath, makedirs, check_type, check_list_depth
from kkpackage.util.logger import set_loglevel, set_logger
logger = set_logger(__name__)


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


class EarlyStoppingError(Exception):
    """early stopping の条件を達成した時に発生する例外"""
    pass


class BaseNN:
    def __init__(
        self,
        # network
        mynn: nn.Module, mtype: str,
        # loss functions
        loss_funcs: List[List[object]]=None, loss_funcs_valid: List[List[object]]=None,
        # optimizer
        optimizer: Optimizer=torch.optim.SGD, optim_dict: dict={"lr":0.001, "weight_decay":0},
        scheduler: _LRScheduler=None ,scheduler_dict: dict=None,
        # train dataloader
        dataloader_train: torch.utils.data.DataLoader=None,
        # validation dataset
        dataloader_valids: List[torch.utils.data.DataLoader]=[],
        # train parameter
        epoch: int=100, batch_size: int=-1, valid_step: int=-1, batch_size_valid: int=-1,
        early_stopping_rounds: int=-1, early_stopping_loss_diff: float=None, 
        # output
        outdir: str="./output_"+datetime.datetime.now().strftime("%Y%m%d%H%M%S"), save_step: int=None
    ):
        """
        Params::
            mynn: PyTorch形式でのNN
            mtype: "cls" or "reg"
            loss_funcs: List[List[Lossfunc]]形式での callable な loss function
            loss_funcs_valid: list形式での callable な loss function. gradientは計算されない. Noneの場合はloss_funcsと同じ
            optimizer: Optimizer を継承した class を set. init で
            dataloader_train: DataLoader形式. 画像系の場合は DataLoader使う
            dataloader_valids: DataLoader形式のList
        """
        # NN
        self.mynn  = mynn
        self.mtype = mtype
        # Loss
        if loss_funcs is not None: check_list_depth(loss_funcs, 2)
        if loss_funcs_valid is not None: check_list_depth(loss_funcs_valid, 2)
        self.loss_funcs       = loss_funcs if loss_funcs is not None else [[]]
        self.loss_funcs_valid = loss_funcs_valid if loss_funcs_valid is not None else loss_funcs
        # Optimizer
        self.optimizer = optimizer(self.mynn.parameters(), **optim_dict)
        self.scheduler = scheduler(self.optimizer , **scheduler_dict) if scheduler is not None else None
        self.optimizer_class = optimizer
        self.optimizer_dict  = optim_dict
        self.scheduler_class = scheduler
        self.scheduler_dict  = scheduler_dict
        # DataLoader
        self.dataloader_train  = dataloader_train
        self.dataloader_valids = dataloader_valids
        # training
        self.batch_size = batch_size
        # validation
        self.valid_step = valid_step
        self.early_stopping_rounds    = early_stopping_rounds
        self.early_stopping_loss_diff = early_stopping_loss_diff
        self.batch_size_valid = batch_size_valid
        # Config
        self.is_cuda   = False
        self.epoch     = epoch
        # Classification
        self.classes_: np.ndarray  = None
        # Other
        self.iter      = 0
        self.iter_best = self.epoch
        self.min_loss_train = float("inf")
        self.min_loss_valid = float("inf")
        self.early_stopping_iter = 0
        self.best_params = {}
        self.outdir = correct_dirpath(outdir)
        makedirs(self.outdir, exist_ok=True, remake=True)
        self.save_step = save_step
        # TensorBoard
        self.writer = SummaryWriter(log_dir=self.outdir + "logs")
        # Check
        self.check_init()

    def __str__(self):
        string = f"""model type     : {self.mtype}
loss functions       : {self.loss_funcs}
loss functions valid : {self.loss_funcs_valid}
optimizer            : {self.optimizer}
batch size           : {self.batch_size}
epoch                : {self.epoch}
network              : {self.mynn}"""
        return string
    
    def check_init(self):
        if self.mtype not in ["cls", "reg"]:
            logger.raise_error(f'"mtype" is "cls" or "reg": {self.mtype}')

    def initialize(self, weight: float=None):
        self.iter      = 0
        self.iter_best = self.epoch
        self.classes_: np.ndarray = None
        self.min_loss_valid  = float("inf")
        self.early_stopping_iter = 0
        self.best_params = {}
        self.mynn.reset_parameters(weight=weight)
        self.optimizer = self.optimizer_class(self.mynn.parameters(), **self.optimizer_dict)
        self.scheduler = self.scheduler_class(self.optimizer , **self.scheduler_dict) if self.scheduler_class is not None else None

    def to_cuda(self):
        """ GPUの使用をONにする """
        self.mynn.to(torch.device("cuda:0")) # moduleがcudaにのっているかどうかは、next(model.parameters()).is_cuda で確認できる
        self.loss_funcs       = [[x.to(torch.device("cuda:0")) for x in listwk] for listwk in self.loss_funcs]
        self.loss_funcs_valid = [[x.to(torch.device("cuda:0")) for x in listwk] for listwk in self.loss_funcs_valid]
        self.is_cuda = True

    def val_to(self, x):
        """ GPUの使用がONであればGPUに乗せる """
        return x.to(torch.device("cuda:0")) if self.is_cuda else x

    def save(self, filename: str=None, is_best: bool=False):
        if is_best:
            if len(self.best_params) > 0:
                torch.save(self.best_params["params"], self.outdir + filename \
                if filename is not None else self.outdir + f'model_best_{self.best_params["iter"]}.pth')
            else:
                logger.warning("self.best_params is nothing.")
                torch.save(self.mynn.state_dict(), self.outdir + filename + f".{self.iter}" if filename is not None else self.outdir + f"model_{self.iter}.pth")
        else:
            torch.save(self.mynn.state_dict(), self.outdir + filename + f".{self.iter}" if filename is not None else self.outdir + f"model_{self.iter}.pth")
    
    def load(self, model_path: str):
        self.mynn.load_state_dict(torch.load(model_path))
        self.mynn.eval()
    
    @classmethod
    def process_data_train_pre(cls, _input): return _input
    @classmethod
    def process_data_train_aft(cls, _input):
        return _input if isinstance(_input, list) or isinstance(_input, tuple) else [_input, ]
    @classmethod
    def process_data_valid_pre(cls, _input): return _input
    @classmethod
    def process_data_valid_aft(cls, _input):
        return _input if isinstance(_input, list) or isinstance(_input, tuple) else [_input, ]
    @classmethod
    def process_label(cls, _input): return _input

    def processes(self, _input: object, label: object=None, is_valid: bool=False):
        """
        processes の IF は、torch.Tensor 以外にも画像なども想定するため、幅広く想定する
        """
        output = None
        if is_valid:
            # pre proc
            output = self.process_data_valid_pre(_input)
            output = self.val_to(output)
            output = self.mynn(output)
            # after proc
            output = self.process_data_valid_aft(output)
        else:
            # pre proc
            output = self.process_data_train_pre(_input)
            output = self.val_to(output)
            output = self.mynn(output)
            # after proc
            output = self.process_data_train_aft(output)
        # label proc
        if label is not None:
            label = self.process_label(label)
            if isinstance(label, list) or isinstance(label, tuple):
                label = [self.val_to(x) for x in label]
            else:
                label = [self.val_to(label), ]
        return output, label
    
    def _train(self, _input: object, label: object):
        self.mynn.train() # train() と eval() は Dropout があるときに区別する必要がある
        self.mynn.zero_grad() # 勾配を初期化
        self.iter += 1
        output, label = self.processes(_input, label=label, is_valid=False)
        # loss calculation
        loss, losses = 0, []
        for i, _loss_funcs in enumerate(self.loss_funcs):
            for loss_func in _loss_funcs:
                losses.append(loss_func(output[i], label[i]))
                loss += losses[-1]
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None: self.scheduler.step()
        loss   = float(loss.to("cpu").detach().item())
        self.min_loss_train = loss
        losses = [float(_x.to("cpu").detach().item()) for _x in losses]
        logger.info(f'iter: {self.iter}, train: {loss}, loss: {losses}, lr: {"No schedule." if self.scheduler is None else self.scheduler.get_last_lr()[0]}')
        # tensor board
        self.writer.add_scalar("train/total_loss", loss, self.iter)
        for i_loss, _loss in enumerate(losses): self.writer.add_scalar(f"train/loss_{i_loss}", _loss, self.iter)
        # save
        if self.save_step is not None and self.save_step > 0 and self.iter % self.save_step == 0:
            self.save()

    def _valid(self, _input, label, i_valid: int=0):
        self.mynn.eval()
        with torch.no_grad():
            output, label = self.processes(_input, label=label, is_valid=True)
            # loss calculation
            loss_valid, losses_valid = 0, []
            for i, _loss_funcs in enumerate(self.loss_funcs_valid):
                for loss_func in _loss_funcs:
                    losses_valid.append(loss_func(output[i], label[i]))
                    loss_valid += losses_valid[-1]
            loss_valid   = float(loss_valid.to("cpu").detach().item())
            losses_valid = [float(_x.to("cpu").detach().item()) for _x in losses_valid]
            logger.info(f'iter: {self.iter}, valid: {loss_valid}, loss: {losses_valid}')
            # tensor board
            self.writer.add_scalar(f"validation{i_valid}/total_loss", loss_valid, self.iter)
            for i_loss, _loss in enumerate(losses_valid): self.writer.add_scalar(f"validation{i_valid}/loss_{i_loss}", _loss, self.iter)
            self.early_stopping_iter += 1
            # early stopping conditions
            bool_store_early_stopping = False
            if   i_valid == 0 and self.early_stopping_loss_diff is None and self.min_loss_valid > loss_valid:
                bool_store_early_stopping = True
            elif i_valid == 0 and isinstance(self.early_stopping_loss_diff, float) and (loss_valid - self.min_loss_train < self.early_stopping_loss_diff):
                bool_store_early_stopping = True
            if bool_store_early_stopping:
                self.min_loss_valid = loss_valid
                self.early_stopping_iter = 0 # iteration を reset
                self.best_params = {
                    "iter": self.iter,
                    "loss_valid": loss_valid,
                    "params": self.mynn.state_dict().copy()
                }
            if isinstance(self.early_stopping_rounds, int) and self.early_stopping_rounds > 0 and self.early_stopping_iter >= self.early_stopping_rounds:
                # early stopping
                raise EarlyStoppingError
    
    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_valid: Tuple[np.ndarray]=None, y_valid: Tuple[np.ndarray]=None):
        """
        numpy ndarray ベースの train
        """
        if self.mtype in ["cls"]:
            self.classes_: np.ndarray = np.sort(np.unique(y_train)).astype(int)
        indexes = np.arange(x_train.shape[0])
        x_train = torch.from_numpy(x_train)
        y_train = torch.from_numpy(y_train)
        if x_valid is not None:
            if isinstance(x_valid, np.ndarray):
                x_valid = [torch.from_numpy(x_valid)]
                y_valid = [torch.from_numpy(y_valid)]
            elif isinstance(x_valid, list) or isinstance(x_valid, tuple):
                x_valid = [torch.from_numpy(_ndf) for _ndf in x_valid]
                y_valid = [torch.from_numpy(_ndf) for _ndf in y_valid]
            indexes_valid = [np.arange(ndf.shape[0]) for ndf in x_valid]
        try:
            for _ in range(self.epoch):
                _index = np.random.permutation(indexes)[:x_train.shape[0] if self.batch_size <= 0 else self.batch_size]
                self._train(x_train[_index], y_train[_index])
                if x_valid is not None and self.valid_step is not None and self.valid_step > 0 and self.iter % self.valid_step == 0:
                    for i, (_x_valid, _y_valid) in enumerate(zip(x_valid, y_valid)):
                        _index = np.random.permutation(indexes_valid[i])[:_x_valid.shape[0] if self.batch_size_valid <= 0 else self.batch_size_valid]
                        self._valid(_x_valid[_index], _y_valid[_index], i_valid=i)
        except EarlyStoppingError:
            logger.warning(f'early stopping. iter: {self.iter}, best_iter: {self.best_params["iter"]}, loss: {self.best_params["loss_valid"]}')
            self.iter_best = self.best_params["iter"]
        self.writer.close()
        self.save(is_best=True)

    def train(self):
        """
        dataloader を使う場合はこっち
        """
        try:
            for _ in range(self.epoch):
                for _input, label in self.dataloader_train:
                    # train
                    self._train(_input, label)
                    # validation
                    if len(self.dataloader_valids) > 0 and self.valid_step is not None and self.valid_step > 0 and self.iter % self.valid_step == 0:
                        for i_valid, dataloader_valid in enumerate(self.dataloader_valids):
                            _input, label = next(iter(dataloader_valid))
                            self._valid(_input, label, i_valid=i_valid)
        except EarlyStoppingError:
            logger.warning(f'early stopping. iter: {self.iter}, best_iter: {self.best_params["iter"]}, loss: {self.best_params["loss_valid"]}')
            self.iter_best = self.best_params["iter"]
        self.writer.close()
        self.save(is_best=True)

    def predict(self, _x: np.ndarray, _y: np.ndarray=None, batch_size: int=-1, out_index: int=0):
        """
        numpy ndarray ベースの predict
        Params::
            batch_size: GPUで計算できないサイズが考えられるので分割できるようにしておく
        """
        if not _y:
            output = self.predict_proba(_x, _y=_y, batch_size=batch_size, out_index=out_index)
        else:
            output, label = self.predict_proba(_x, _y=_y, batch_size=batch_size, out_index=out_index)
        if self.mtype in ["cls"]:
            output = np.argmax(output, axis=1)
        if _y is None:
            return output
        else:
            return output, label
    
    def predict_proba(self, _x: np.ndarray, _y: np.ndarray=None, batch_size: int=-1, out_index: int=0):
        """
        numpy ndarray ベースの predict
        """
        if self.mtype not in ["cls"]: logger.raise_error(f'model type is not cls. {self.mtype}')
        _x = torch.from_numpy(_x)
        if _y is not None: _y = torch.from_numpy(_y)
        self.mynn.eval()
        output, label = None, None
        with torch.no_grad():
            if batch_size > 1:
                listwk = divide_index(np.arange(_x.shape[0]), n_div=batch_size)
                output, label = [], []
                for _index in listwk:
                    _output, _label = self.processes(_x[_index], label=(None if _y is None else _y[_index]), is_valid=True)
                    _output = _output[out_index].to("cpu").detach().numpy()
                    output.append(_output)
                    if _y is not None:
                        _label = _label[out_index].to("cpu").detach().numpy()
                        label.append(_label)
                output = np.concatenate(output, axis=0)
                if _y is not None: label = np.concatenate(label, axis=0)
                else: label = None
            else:
                output, label = self.processes(_x, label=_y, is_valid=True)
                output = output[out_index].to("cpu").detach().numpy()
                if _y is not None: label = label.to("cpu").detach().numpy()
        if self.mtype not in ["reg"]:
            if len(output.shape) == 2 and output.shape[1] == 1:
                # regression で 1次元の場合はreshapeする
                output = output.reshape(-1)
            else:
                pass
        elif self.mtype in ["cls"]:
            if   len(output.shape) == 2 and output.shape[1] == 1:
                # binary class
                output = np.concatenate([1 - output, output], axis=1)
            elif len(output.shape) == 2 and output.shape[1] >  1:
                # multi class
                pass
            else:
                logger.raise_error(f"output: {output.shape} is not expected.")
        if _y is None:
            return output
        else:
            return output, label

import os, random, datetime
from typing import List, Tuple
import numpy as np
import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter

# local package
from kkutils.util.dataframe import divide_index
from kkutils.util.numpy import softmax
from kkutils.util.com import is_callable, correct_dirpath, makedirs, check_type, check_list_depth, set_logger
logger = set_logger(__name__)


__all__ = [
    "BaseNN",
    "EarlyStoppingError",
]


class EarlyStoppingError(Exception):
    """early stopping の条件を達成した時に発生する例外"""
    pass


class BaseNN:
    def __init__(
        self,
        # network
        mynn: nn.Module, mtype: str="cls",
        # loss functions
        loss_funcs: List[object]=None, loss_funcs_valid: List[object]=None,
        # optimizer
        optimizer: Optimizer=torch.optim.SGD, optim_params: dict={"lr":0.001, "weight_decay":0},
        scheduler: _LRScheduler=None ,scheduler_params: dict=None,
        # train dataloader
        dataloader_train: torch.utils.data.DataLoader=None,
        # validation dataset
        dataloader_valids: List[torch.utils.data.DataLoader]=[],
        # train parameter
        epoch: int=100, batch_size: int=-1, valid_step: int=-1, batch_size_valid: int=-1,
        early_stopping_rounds: int=-1, early_stopping_loss_diff: float=None, 
        # output
        outdir: str="./output_"+datetime.datetime.now().strftime("%Y%m%d%H%M%S"), save_step: int=None,
        # others
        random_seed: int=0, num_workers: int=1
    ):
        """
        Params::
            mynn: PyTorch形式でのNN
            mtype: "cls" or "reg"
            loss_funcs: List[List[Lossfunc]]形式での callable な loss function
                ListのList[Lossfunc]になっている。2階層目は、outputが複数に分岐して別々にlossを計算する場合. 1階層目は別々のlossを計算する場合
            loss_funcs_valid: list形式での callable な loss function. gradientは計算されない. Noneの場合はloss_funcsと同じ
            optimizer: Optimizer を継承した class を set. init で
            dataloader_train: DataLoader形式. 画像や言語系の場合は DataLoader使う
            dataloader_valids: DataLoader形式のList
        """
        # NN
        self.mynn  = mynn
        self.mtype = mtype
        # Loss
        if loss_funcs is not None: check_list_depth(loss_funcs, 1)
        if loss_funcs_valid is not None: check_list_depth(loss_funcs_valid, 1)
        self.loss_funcs       = loss_funcs if loss_funcs is not None else []
        self.loss_funcs_valid = loss_funcs_valid if loss_funcs_valid is not None else self.loss_funcs
        # Optimizer
        self.optimizer = optimizer(self.mynn.parameters(), **optim_params)
        self.scheduler = scheduler(self.optimizer , **scheduler_params) if scheduler is not None else None
        self.optimizer_class  = optimizer
        self.optimizer_params = optim_params
        self.scheduler_class  = scheduler
        self.scheduler_params = scheduler_params
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
        self.set_seed_all(random_seed)
        self.num_workers = num_workers
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
    
    def set_seed_all(self, seed: int):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info('Set random seeds')

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
        self.optimizer = self.optimizer_class(self.mynn.parameters(), **self.optimizer_params)
        self.scheduler = self.scheduler_class(self.optimizer , **self.scheduler_params) if self.scheduler_class is not None else None

    def to_cuda(self):
        """ GPUの使用をONにする """
        self.mynn.to(torch.device("cuda:0")) # moduleがcudaにのっているかどうかは、next(model.parameters()).is_cuda で確認できる
        self.loss_funcs       = [x.to(torch.device("cuda:0")) for x in self.loss_funcs]
        self.loss_funcs_valid = [x.to(torch.device("cuda:0")) for x in self.loss_funcs_valid]
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
        for i, loss_func in enumerate(self.loss_funcs):
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
            for i, loss_func in enumerate(self.loss_funcs_valid):
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

    def predict(
        self, _x: np.ndarray=None, _y: np.ndarray=None, 
        dataloader: torch.utils.data.DataLoader=None, is_label: bool=False, 
        batch_size: int=-1, out_index: int=0
    ):
        """
        predict処理.
        詳細は predict_proba を参照
        """
        if (_x is not None and _y is None) or (dataloader is not None and is_label == False):
            output        = self.predict_proba(_x=_x, _y=_y, dataloader=dataloader, is_label=is_label, batch_size=batch_size, out_index=out_index)
        else:
            output, label = self.predict_proba(_x=_x, _y=_y, dataloader=dataloader, is_label=is_label, batch_size=batch_size, out_index=out_index)
        if self.mtype in ["cls"]:
            output = np.argmax(output, axis=1)
        if (_x is not None and _y is None) or (dataloader is not None and is_label == False):
            return output
        else:
            return output, label
    
    def predict_proba(
        self, _x: np.ndarray=None, _y: np.ndarray=None, batch_size: int=-1, 
        dataloader: torch.utils.data.DataLoader=None, is_label: bool=False, 
        out_index: int=0, is_softmax: bool=False
    ):
        """
        cls と reg のときの predict処理共通実装. cls の場合のラベル推定はpredict で処理する.
        Params::
            _x: input. numpy 形式
            _y: label. あれば入力
            batch_size: numpy 形式の場合のbatch size
            dataloader: 画像や言語など、dataloader形式の方が楽な場合はこっちに入力する
            is_label: dataloader形式の場合の正解ラベルがあるかどうか. ※ただし、for _input, label in dataloader のように出力は２つ用意する
            out_index: loss を複数計算している場合など、outputが複数の場合に、どの要素番号を採用するか
        Output::
            label があるかどうかで、出力の数が変わる
            np.ndarray or (np.ndarray, np.ndarray, )
        """
        self.mynn.eval()
        with torch.no_grad():
            if dataloader is not None:
                output, label = self.predict_proba_dataloader(dataloader=dataloader, is_label=is_label, out_index=out_index)
            else:
                output, label = self.predict_proba_numpy(_x=_x, _y=_y, batch_size=batch_size, out_index=out_index)
        if self.mtype in ["reg"]:
            if len(output.shape) == 2 and output.shape[1] == 1:
                # regression で 1次元の場合はreshapeする
                output = output.reshape(-1)
            else:
                pass
        elif self.mtype in ["cls"]:
            if is_softmax:
                output = softmax(output)
            if   len(output.shape) == 2 and output.shape[1] == 1:
                # binary class
                output = np.concatenate([1 - output, output], axis=1)
            elif len(output.shape) == 2 and output.shape[1] >  1:
                # multi class
                pass
            else:
                logger.raise_error(f"output: {output.shape} is not expected.")
        if (_x is not None and _y is None) or (dataloader is not None and is_label == False):
            return output
        else:
            return output, label

    def predict_proba_numpy(self, _x: np.ndarray, _y: np.ndarray=None, batch_size: int=-1, out_index: int=0):
        """
        numpy ndarray ベースの predict
        """
        _x = torch.from_numpy(_x)
        if _y is not None: _y = torch.from_numpy(_y)
        output, label = None, None
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
        return output, label

    def predict_proba_dataloader(self, dataloader: torch.utils.data.DataLoader, is_label: bool=False, out_index: int=0):
        """
        dataloader ベースの predict
        """
        output, label = [], []
        for _input, _label in dataloader:
            _output, _label = self.processes(_input, label=_label if is_label else None, is_valid=True)
            _output = _output[out_index].to("cpu").detach().numpy()
            output.append(_output)
            if is_label:
                _label = _label[out_index].to("cpu").detach().numpy()
                label.append(_label)
        output = np.concatenate(output, axis=0)
        if is_label: label = np.concatenate(label, axis=0)
        return output, label


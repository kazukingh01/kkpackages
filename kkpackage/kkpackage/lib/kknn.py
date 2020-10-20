import datetime
from typing import List
from collections import namedtuple
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
# local package
from kkpackage.util.common import is_callable, correct_dirpath, makedirs
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
                    self.add_module(name, layer.module(*layer.params, **layer.kwards))
            elif layer.node == 0:
                self.add_module(name, layer.module(in_size, *layer.params, **layer.kwards))
            else:
                self.add_module(name, layer.module(in_size, layer.node, *layer.params, **layer.kwards))
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
            if   self.indexes[i] is None:
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



class BaseNN:
    def __init__(
        self,
        # network
        mynn: nn.Module, 
        # train dataset
        dataset_train: torch.utils.data.Dataset,
        # train dataloader
        num_workers: int=1, batch_size: int=2, 
        # validation dataset
        dataset_valids: List[torch.utils.data.Dataset]=[],
        # validation dataloader
        valid_step: int=None, batch_size_valid: int=2, 
        # optimizer
        lr: float=0.001, epoch: int=100, 
        # output
        outdir: str="./output_"+datetime.datetime.now().strftime("%Y%m%d%H%M%S"), save_step: int=50
    ):
        # NN
        self.mynn = mynn
        # optimizer
        self.optimizer = nn.optim.RAdam(self.mynn.parameters(), lr=lr, weight_decay=0)
        # DataLoader
        self.dataloader_train = torch.utils.data.DataLoader(
            dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, 
            drop_last=True, collate_fn=self.collate_fn
        )
        self.dataloader_valids = []
        for dataset_valid in dataset_valids:
            self.dataloader_valids.append(
                torch.utils.data.DataLoader(
                    dataset_valid, batch_size=batch_size_valid, shuffle=True, num_workers=num_workers, 
                    drop_last=True, collate_fn=partial(self.collate_fn, is_train=False)
                )
            )
        # Process
        self.process_data_train_pre = []
        self.process_data_train_aft = []
        self.process_data_valid_pre = []
        self.process_data_valid_aft = []
        self.process_label          = []
        # Loss
        self.loss_funcs = []
        # validation
        self.valid_step = valid_step
        # Config
        self.is_cuda  = False
        self.epoch    = epoch
        # Other
        self.iter     = 0
        self.min_loss = float("inf")
        self.best_params = {}
        self.outdir = correct_dirpath(outdir)
        makedirs(self.outdir, exist_ok=True, remake=True)
        self.save_step = save_step
        # TensorBoard
        self.writer = SummaryWriter(log_dir=self.outdir + "logs")

    def to_cuda(self):
        """ GPUの使用をONにする """
        self.mynn.to(torch.device("cuda:0")) # moduleがcudaにのっているかどうかは、next(model.parameters()).is_cuda で確認できる
        self.is_cuda = True

    def val_to(self, x):
        """ GPUの使用がONであればGPUに乗せる """
        return x.to(torch.device("cuda:0")) if self.is_cuda else x

    @classmethod
    def collate_fn(cls, batch, is_train: bool=True) -> (List[object], List[object], ):
        """
        以下のように実装する
        Usage::
            images, labels = [], []
            for image, label in batch:
                images.append(image)
                labels.append(label)
            return images, labels
        """
        raise NotImplementedError()

    def save(self, filename: str=None, is_best: bool=False):
        if is_best:
            if len(self.best_params) > 0:
                torch.save(self.best_params["params"], self.outdir + filename + f'.{self.best_params["iter"]}' \
                if filename is not None else self.outdir + f'model_best_{self.best_params["iter"]}.pth')
            else:
                logger.raise_error("self.best_params is nothing.")
        else:
            torch.save(self.mynn.state_dict(), self.outdir + filename + f".{self.iter}" if filename is not None else self.outdir + f"model_{self.iter}.pth")
    
    def load(self, model_path: str):
        self.mynn.load_state_dict(torch.load(model_path))
        self.mynn.eval()
    
    def predict(self, _input):
        self.mynn.eval()
        with torch.no_grad():
            # pre proc
            for _proc in self.process_data_train_pre:
                _input = _proc(_input)
            output = self.val_to(_input)
            output = self.mynn(output)
            # after proc
            for _proc in self.process_data_train_aft:
                output = _proc(output)
        return output
    
    def train(self):
        self.mynn.train() # train() と eval() は Dropout があるときに区別する必要がある
        for _ in range(self.epoch):
            for _input, label in self.dataloader_train:
                self.iter += 1
                self.mynn.zero_grad() # 勾配を初期化
                # pre proc
                for _proc in self.process_data_train_pre: _input = _proc(_input)
                for _proc in self.process_label:          label  = _proc(label)
                label  = self.val_to(label)
                output = self.val_to(_input)
                output = self.mynn(output)
                # after proc
                for _proc in self.process_data_train_aft: output = _proc(output)
                print((output > 0.5).sum())
                # loss calculation
                loss, losses = 0, []
                for loss_func in self.loss_funcs:
                    losses.append(loss_func(output, label))
                    loss += losses[-1]
                loss.backward()
                self.optimizer.step()
                loss   = float(loss.to("cpu").detach().item())
                losses = [float(_x.to("cpu").detach().item()) for _x in losses]
                logger.info(f'iter: {self.iter}, train: {loss}, loss: {losses}')
                # tensor board
                self.writer.add_scalar("train/total_loss", loss, self.iter)
                for i_loss, _loss in enumerate(losses): self.writer.add_scalar(f"train/loss_{i_loss}", _loss, self.iter)
                # save
                if self.iter % self.save_step == 0:
                    self.save()
                # validation
                if len(self.dataloader_valids) > 0 and self.valid_step is not None and self.iter % self.valid_step == 0:
                    self.mynn.eval()
                    with torch.no_grad():
                        for i_valid, dataloader_valid in enumerate(self.dataloader_valids):
                            _input, label = next(iter(dataloader_valid))
                            # pre proc
                            for _proc in self.process_data_valid_pre: _input = _proc(_input)
                            for _proc in self.process_label:          label  = _proc(label)
                            label  = self.val_to(label)
                            output = self.val_to(_input)
                            output = self.mynn(output)
                            # after proc
                            for _proc in self.process_data_valid_aft: output = _proc(output)
                            # loss calculation
                            loss_valid, losses_valid = 0, []
                            for loss_func in self.loss_funcs:
                                losses_valid.append(loss_func(output, label))
                                loss_valid += losses_valid[-1]
                            loss_valid   = float(loss_valid.to("cpu").detach().item())
                            losses_valid = [float(_x.to("cpu").detach().item()) for _x in losses_valid]
                            logger.info(f'iter: {self.iter}, valid: {loss_valid}, loss: {losses_valid}')
                            # tensor board
                            self.writer.add_scalar(f"validation{i_valid}/total_loss", loss_valid, self.iter)
                            for i_loss, _loss in enumerate(losses_valid): self.writer.add_scalar(f"validation{i_valid}/loss_{i_loss}", _loss, self.iter)
                            if i_valid == 0 and self.min_loss > loss:
                                self.min_loss = loss
                                self.best_params = {
                                    "iter": self.iter,
                                    "loss_train": loss,
                                    "loss_valid": loss_valid,
                                    "params": self.mynn.state_dict().copy()
                                }
                    self.mynn.train()
        self.writer.close()
        self.save(is_best=True)

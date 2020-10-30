import os, json, random, datetime
import numpy as np
import cv2
from typing import List, Tuple
import torch
from torch import nn
from torchvision import models
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
from functools import partial
from RandAugment import RandAugment
import torch_optimizer as optim
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter

# local package
from imageaug import AugHandler, Augmenter as aug
from kkpackage.lib.kknn import TorchNN, Layer
from kkimagemods.util.images import pil2cv, cv2pil
from kkimagemods.util.dataframes import split_data_balance
from kkimagemods.util.common import get_file_list, correct_dirpath, makedirs
from kkimagemods.util.logger import set_logger, set_loglevel
logger = set_logger()



class ImageNN(nn.Module):
    """
    model_zoo を流用してファインチューニングを行う NN の定義
    """
    def __init__(self):
        super().__init__()
        nn_trained = models.densenet121(pretrained=True)
        torch_nn_list = []
        for _ in range(8):
            torch_nn_list.append(
                TorchNN(
                    128,
                    Layer("", nn.Linear, 64,   None, (), {}),
                    Layer("", nn.ReLU,   None, None, (), {}),
                    Layer("", nn.Linear, 32,   None, (), {}),
                    Layer("", nn.ReLU,   None, None, (), {}),
                    Layer("", nn.Linear, 16,   None, (), {}),
                    Layer("", nn.ReLU,   None, None, (), {}),
                    Layer("", nn.Linear, 1,    None, (), {}),
                    Layer("", nn.Sigmoid,None, None, (), {}),
                )
            )
        torch_nn_output1 = TorchNN(
            128,
            Layer("", nn.Linear, 64,   None, (), {}),
            Layer("", nn.ReLU,   None, None, (), {}),
            Layer("", nn.Linear, 32,   None, (), {}),
            Layer("", nn.ReLU,   None, None, (), {}),
            Layer("", nn.Linear, 16,   None, (), {}),
            Layer("", nn.ReLU,   None, None, (), {}),
            Layer("", nn.Linear, 1,    None, (), {}),
            Layer("", nn.Sigmoid,None, None, (), {}),
        )
        torch_nn_output2 = TorchNN(
            128,
            Layer("", nn.Linear, 64,   None, (), {}),
            Layer("", nn.ReLU,   None, None, (), {}),
            Layer("", nn.Linear, 32,   None, (), {}),
            Layer("", nn.ReLU,   None, None, (), {}),
            Layer("", nn.Linear, 16,   None, (), {}),
            Layer("", nn.ReLU,   None, None, (), {}),
            Layer("", nn.Linear, 4,    None, (), {}),
            Layer("", nn.Softmax,None, None, (), {"dim":1}),
        )
        torch_nn = TorchNN(
            1024,
            Layer("", nn.Conv2d,       512,    None, (), {"kernel_size":1, "stride":1,}),
            Layer("", nn.BatchNorm2d,  0,      None, (), {"eps":1e-05, "momentum":0.1, "affine":True, "track_running_stats":True}),
            Layer("", nn.ReLU,         None,   None, (), {}),        
            Layer("", nn.Conv2d,       256,    None, (), {"kernel_size":1, "stride":1,}),
            Layer("", nn.BatchNorm2d,  0,      None, (), {"eps":1e-05, "momentum":0.1, "affine":True, "track_running_stats":True}),
            Layer("", nn.ReLU,         None,   None, (), {}),        
            Layer("", nn.Conv2d,       128,    None, (), {"kernel_size":1, "stride":1,}),
            Layer("", nn.BatchNorm2d,  0,      None, (), {"eps":1e-05, "momentum":0.1, "affine":True, "track_running_stats":True}),
            Layer("", nn.ReLU,         None,   None, (), {}),        
            Layer("", nn.Conv2d,       64,     None, (), {"kernel_size":1, "stride":1,}),
            Layer("", nn.BatchNorm2d,  0,      None, (), {"eps":1e-05, "momentum":0.1, "affine":True, "track_running_stats":True}),
            Layer("", nn.ReLU,         None,   None, (), {}),
            Layer("", nn.Identity,     64*7*7, "reshape(x,-1)", (), {}),
            Layer("", nn.Linear,       2048,   None, (), {}),
            Layer("", nn.BatchNorm1d,  0,      None, (), {}),
            Layer("", nn.Dropout,      None,   None, (), {"p":0.4}),
            Layer("", nn.ReLU,         None,   None, (), {}),
            Layer("", nn.Linear,       1024,   None, (), {}),
            Layer("", nn.BatchNorm1d,  0,      None, (), {}),
            Layer("", nn.Dropout,      None,   None, (), {"p":0.4}),
            Layer("", nn.ReLU,         None,   None, (), {}),
            Layer("", nn.Linear,       516,    None, (), {}),
            Layer("", nn.BatchNorm1d,  0,      None, (), {}),
            Layer("", nn.Dropout,      None,   None, (), {"p":0.25}),
            Layer("", nn.ReLU,         None,   None, (), {}),
            Layer("", nn.Linear,       256,    None, (), {}),
            Layer("", nn.BatchNorm1d,  0,      None, (), {}),
            Layer("", nn.Dropout,      None,   None, (), {"p":0.25}),
            Layer("", nn.ReLU,         None,   None, (), {}),
            Layer("", nn.Linear,       128,    None, (), {}),
            Layer("", nn.BatchNorm1d,  0,      None, (), {}),
            Layer("", nn.ReLU,         None,   None, (), {}),
            #Layer("", nn.Identity,     None,   "split_8", (), {}),
            Layer("", torch_nn_output1,None,   None, (), {}),
            #Layer("", nn.Identity,     None,   "out_split", (), {}),
        )
        self.add_module("model_zoo", nn_trained.__getattr__("features")) # ImageNetの画像で学習したWeightを使える
        self.add_module("my_nn",     torch_nn)


    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = inputs.clone()
        output = self.__getattr__("model_zoo")(output)
        output = self.__getattr__("my_nn")(output)
        return output



class MyDataset(torch.utils.data.Dataset):
    def __init__(
        self, root_dirpath: str, json_label: str,
        transforms: List[object]=None, 
    ):
        super().__init__()
        """
        label infomation load
        Format::
            {
                "test0.png": 1,  # or [1,2,3]. int や str, List で記述する
                "test1.png": 1,
                "test2.png": 1,
                ...
            }
        """
        self.json_label = json.load(open(json_label)) if type(json_label) == str else json_label
        self.json_label = {x:tuple(y) if type(y) in [list, tuple] else (y,) for x, y in self.json_label.items()} # 全てTupleに変換しておく
        self.root_dirpath = correct_dirpath(root_dirpath)
        self.image_names  = [x for x in self.json_label.keys()]
        self.transforms   = transforms
        self.len          = len(self.image_names)


    def __len__(self):
        return self.len


    def __getitem__(self, index):
        name   = self.image_names[index]
        img    = Image.open(self.root_dirpath + name)
        img    = self.transforms(img)
        labels = self.json_label[name]
        return img, labels



class MyClassifier:
    """
    分類機. Dataloader など実装
    """
    def __init__(
        self, 
        # network
        mynn: nn.Module, 
        # dataset
        root_dirpath: str, json_path: str, 
        # optimizer, batch
        lr: float=0.001, batch_size: int=2, num_workers: int=1, epoch: int=100, 
        # validation
        validation_samples: float=-1, json_valid_paths: dict={}, batch_size_valid: int=1, valid_step: int=10,
        # output
        outdir: str="./output_"+datetime.datetime.now().strftime("%Y%m%d%H%M%S"), save_step: int=50
    ):
        # NN
        self.mynn = mynn
        # optimizer
        self.optimizer = optim.RAdam(self.mynn.parameters(), lr=lr, weight_decay=0)
        # Loss Function
        self.loss_funcs = [
            nn.BCELoss(),
            #nn.CrossEntropyLoss(),
            #nn.SmoothL1Loss(),
        ]
        self.loss_preprocs = [
            lambda x: x.to(torch.float32),
            #lambda x: x.to(torch.long),
        ]
        # Transform
        self.preprocess_img = transforms.Compose([
            MyResize(224),
            transforms.CenterCrop(224),
        ])
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            lambda x: x[:3],
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # default augmentations
        self.augmentations = transforms.Compose([
            MyRandAugment(1, 1),
            pil2cv,
            RandomRotation90(),
            RandomFliplr(),
            cv2pil,
        ])

        # DataSet
        self.mydataset = MyDataset(root_dirpath, json_path, transforms=self.transform)
        # Validation
        self.valid_step    = valid_step
        self.is_validation = True if (type(validation_samples) in [int,float] and validation_samples > 0) or len(json_valid_paths) > 0 else False
        self.dataloaders_valid: OrderedDict = OrderedDict()
        dataset_train, dataset_valid = self.mydataset, None
        if self.is_validation:
            transform_valid = partial(self.transform, is_train=False)
            if (type(validation_samples) in [int,float] and validation_samples > 0):
                listwk = list(self.mydataset.json_label.keys())
                validation_samples = int(len(listwk) * validation_samples)
                listwk = np.random.permutation(listwk)
                samples_train   = listwk[:validation_samples ]
                samples_valid   = listwk[ validation_samples:]
                dataset_train   = MyDataset(root_dirpath, {x:self.mydataset.json_label[x] for x in samples_train}, transforms=self.transform)
                dataset_valid   = MyDataset(root_dirpath, {x:self.mydataset.json_label[x] for x in samples_valid}, transforms=transform_valid)
                # Train data split
                self.dataloaders_valid["normal_validation"] = torch.utils.data.DataLoader(
                    dataset_valid, batch_size=batch_size_valid, shuffle=True, num_workers=num_workers, 
                    drop_last=True, collate_fn=self.collate_fn
                )
            # Custom validation dataset
            for i_valid, (valid_dirpath, json_valid_path, ) in enumerate(json_valid_paths.items()):
                dataset_valid = MyDataset(correct_dirpath(valid_dirpath), json_valid_path, transforms=transform_valid)
                self.dataloaders_valid["custom_validation_"+str(i_valid)] = torch.utils.data.DataLoader(
                    dataset_valid, batch_size=batch_size_valid, shuffle=True, num_workers=num_workers, 
                    drop_last=True, collate_fn=self.collate_fn
                )
        # Train DataLoader
        self.dataloader_train = torch.utils.data.DataLoader(
            dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, 
            drop_last=True, collate_fn=self.collate_fn
        )
        # Config
        self.is_cuda  = False
        self.epoch    = epoch
        # Other
        self.iter     = 0
        self.min_loss = np.inf
        self.best_params = {}
        self.outdir = correct_dirpath(outdir)
        makedirs(self.outdir, exist_ok=True, remake=True)
        self.save_step = save_step
        # TensorBoard
        self.writer = SummaryWriter(log_dir=self.outdir + "logs")


    @classmethod
    def collate_fn(cls, batch, to_tensor=True):
        images, labels = [], []
        for image, label in batch:
            images.append(image)
            labels.append(label)
        if to_tensor:
            images = torch.cat([x.unsqueeze(0) for x in images], axis=0)
            labels = torch.Tensor(labels)
        return images, labels
    

    def samples_dataloader(self, filename: str=None, sample_size: int=1, preview: bool=False, save: bool=False):
        sample_collate_fn  = partial(self.collate_fn, to_tensor=False)
        sample_transformer = partial(self.transform, is_train=True, to_tensor=False)
        sample_json_label  = self.mydataset.json_label
        if filename is not None:
            label = sample_json_label[filename]
            sample_json_label = {}
            sample_json_label[filename] = label
        sample_dataloader  = torch.utils.data.DataLoader(
            MyDataset(self.mydataset.root_dirpath, sample_json_label, transforms=sample_transformer), 
            batch_size=1, shuffle=True, num_workers=1, drop_last=True, collate_fn=sample_collate_fn
        )
        imgs, labels = [], []
        while(len(imgs) < sample_size):
            for _imgs, _labels in sample_dataloader:
                for img   in _imgs  : imgs.append(img)
                for label in _labels: labels.append(label)
                if len(imgs) < sample_size:
                    break
        imgs   = imgs[:sample_size]
        labels = labels[:sample_size]
        if preview:
            if sample_size > 10:
                logger.warning("max preview images is 10.")
            for img in imgs[:10]:
                if type(img) == np.ndarray:
                    cv2.imshow("test", img)
                    cv2.waitKey(0)
                else:
                    img.show()
        if save:
            makedirs(self.outdir + "sample_augmentation/", exist_ok=True, remake=True)
            for i, img in enumerate(imgs):
                out_fnema = self.outdir + f"sample_augmentation/samples_dataloader_{i}.png"
                if type(img) == np.ndarray:
                    cv2.imwrite(out_fnema, img)
                else:
                    img.save(out_fnema)
        return imgs, labels


    def transform(self, img: Image, is_train: bool=True, to_tensor: bool=True) -> torch.Tensor:
        if is_train:
            img = self.augmentations(img)
        img = self.preprocess_img(img)
        if to_tensor == False: return img
        # ToTensor は PIL or np.ndarray で変換可能
        output = self.preprocess(img)
        return output
    

    def to_cuda(self):
        """ GPUの使用をONにする """
        self.mynn.to(torch.device("cuda:0")) # moduleがcudaにのっているかどうかは、next(model.parameters()).is_cuda で確認できる
        self.is_cuda = True


    def val_to(self, x):
        """ GPUの使用がONであればGPUに乗せる """
        return x.to(torch.device("cuda:0")) if self.is_cuda else x


    def train(self):
        self.mynn.train() # train() と eval() は Dropout があるときに区別する必要がある
        for _ in range(self.epoch):
            for img, label in self.dataloader_train:
                self.iter += 1
                self.mynn.zero_grad() # 勾配を初期化
                img    = self.val_to(img)
                label  = self.val_to(label)
                output = self.mynn(img)
                loss   = 0
                losses = []
                if len(self.loss_funcs) > 1:
                    for i_loss, loss_func in enumerate(self.loss_funcs):
                        _loss = loss_func(output[i_loss], self.loss_preprocs[i_loss](label[:, i_loss]))
                        loss += _loss
                        losses.append(_loss.to("cpu").detach().item())
                else:
                    loss = self.loss_funcs[0](output, self.loss_preprocs[0](label[:, 0]))
                loss.backward()
                self.optimizer.step()
                loss_train, loss_valid = np.nan, np.nan
                loss_train = float(loss.to("cpu").detach().item())
                logger.info(f'iteration: {self.iter}, total loss: {loss_train}, loss: {losses}')
                self.writer.add_scalar("train/total_loss", loss_train, self.iter)
                for i_loss in range(len(losses)): self.writer.add_scalar(f"train/loss_{i_loss}", losses[i_loss], self.iter)
                # save
                if self.iter % self.save_step == 0:
                    self.save()
                # validation
                if self.is_validation and self.iter % self.valid_step == 0:
                    self.mynn.eval()
                    with torch.no_grad():
                        for valid_name, dataloader_valid in self.dataloaders_valid.items():
                            img, label = next(iter(dataloader_valid))
                            img    = self.val_to(img)
                            label  = self.val_to(label)
                            output = self.mynn(img)
                            loss_valid   = 0
                            losses_valid = []
                            if len(self.loss_funcs) > 1:
                                for i_loss, loss_func in enumerate(self.loss_funcs):
                                    _loss = loss_func(output[i_loss], self.loss_preprocs[i_loss](label[:, i_loss]))
                                    loss_valid += _loss
                                    losses_valid.append(_loss.to("cpu").detach().item())
                            else:
                                loss_valid = self.loss_funcs[0](output, self.loss_preprocs[0](label[:, 0]))
                            loss_valid = float(loss_valid.to("cpu").detach().item())
                            if self.min_loss > loss_valid:
                                self.min_loss = loss_valid
                                self.best_params = {
                                    "iter": self.iter,
                                    "loss_train": loss_train,
                                    "loss_valid": loss_valid,
                                    "params": self.mynn.state_dict().copy()
                                }
                            logger.info(f'total loss {valid_name}: {loss_valid}, losses valid: {losses_valid}')
                            self.writer.add_scalar(f"validation/total_valid_loss_{valid_name}", loss_valid, self.iter)
                            for i_loss in range(len(losses_valid)): self.writer.add_scalar(f"validation/{valid_name}_loss_{i_loss}", losses_valid[i_loss], self.iter)
                    self.mynn.train()
        self.writer.close()
        self.save(is_best=True)


    def predict(self, imgpath: str):
        self.mynn.eval()
        with torch.no_grad():
            img  = Image.open(imgpath)
            tens = self.transform(img, is_train=False, to_tensor=True)
            tens = tens.unsqueeze(0)
            tens = self.val_to(tens)
            output = self.mynn(tens)
            return output
    

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
        self.optimizer = optim.RAdam(self.mynn.parameters(), lr=lr, weight_decay=0)
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



class MyAutoEncoder(BaseNN):
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
        super().__init__(
            mynn, dataset_train,
            num_workers=num_workers, batch_size=batch_size,
            dataset_valids=dataset_valids,
            valid_step=valid_step, batch_size_valid=batch_size_valid, 
            lr=lr, epoch=epoch, outdir=outdir, save_step=save_step
        )
        self.process_data_train_aft = [
            lambda x: x.reshape(-1),
        ]
        self.process_data_valid_aft = [
            lambda x: x.reshape(-1),
        ]
        self.process_label = [
            lambda x: x.to(torch.float32),
            lambda x: x.reshape(-1),
        ]
        self.loss_funcs = [
            nn.BCELoss()
        ]

    @classmethod
    def collate_fn(cls, batch, is_train: bool=False):
        data = []
        for _data, _ in batch:
            data.append(_data)
        data = torch.cat([x.unsqueeze(0) for x in data], axis=0)
        return data, data

    def predict(self, _input, all_layer: bool=True):
        self.mynn.eval()
        with torch.no_grad():
            # pre proc
            for _proc in self.process_data_train_pre:
                _input = _proc(_input)
            output = self.val_to(_input)
            if all_layer:
                output = self.mynn(output)
            else:
                output = self.mynn.encoder(output)
        return output

"""
Augmentations
"""
class RandomRotation90(object):
    """Rotate by one of the given angles."""
    def __init__(self):
        self.values = [0, 1, 2, 3]
    def __call__(self, img: np.ndarray):
        value = random.choice(self.values)
        return np.rot90(img, value) if value > 0 else img

class RandomFliplr(object):
    def __init__(self):
        self.values = [False, True]
    def __call__(self, img: np.ndarray):
        value = random.choice(self.values)
        return np.fliplr(img) if value else img

class MyResize(object):
    def __init__(self, max_size: int):
        self.max_size = max_size
    def __call__(self, img: Image):
        w, h = img.size
        w, h = (self.max_size, self.max_size * h / w, ) if w > h else (self.max_size * w / h, self.max_size, )
        w, h = int(w), int(h)
        return transforms.Resize((h,w))(img)

class MyRandAugment(RandAugment):
    def __init__(self, n, m):
        super().__init__(n, m)
    def __call__(self, img: Image) -> Image:
        img_cp = None
        while(True):
            img_cp = img.copy()
            img_cp = super().__call__(img_cp)
            if np.unique(pil2cv(img_cp)).shape[0] > 3:
                # ために真っ黒な画像になるため、３色以下の場合は操作をやり直す
                break
        return img_cp



"""
Transformation
"""
class MyCenterCrop:
    def __init__(self, size: Tuple[int], padding: object=False):
        self.size = size
        self.padding = padding
    def __call__(self, tens: torch.Tensor)-> torch.Tensor:
        tens_new = torch.zeros(*self.size, dtype=tens.dtype)
        tens_new[:] = self.padding # padding の値で変換
        center_tens_b = [x // 2 for x in tens.shape]
        center_tens_a = [x // 2 for x in tens_new.shape]
        center_width  = [y if x > y else x for x, y in zip(center_tens_b, center_tens_a)]
        if   len(self.size) == 3:
            tens_new[
                center_tens_a[0] - center_width[0]:center_tens_a[0] + center_width[0],
                center_tens_a[1] - center_width[1]:center_tens_a[1] + center_width[1],
                center_tens_a[2] - center_width[2]:center_tens_a[2] + center_width[2],
            ] = tens[
                center_tens_b[0] - center_width[0]:center_tens_b[0] + center_width[0],
                center_tens_b[1] - center_width[1]:center_tens_b[1] + center_width[1],
                center_tens_b[2] - center_width[2]:center_tens_b[2] + center_width[2],
            ]
        elif len(self.size) == 2:
            tens_new[
                center_tens_a[0] - center_width[0]:center_tens_a[0] + center_width[0],
                center_tens_a[1] - center_width[1]:center_tens_a[1] + center_width[1],
            ] = tens[
                center_tens_b[0] - center_width[0]:center_tens_b[0] + center_width[0],
                center_tens_b[1] - center_width[1]:center_tens_b[1] + center_width[1],
            ]
        elif len(self.size) == 1:
            tens_new[
                center_tens_a[0] - center_width[0]:center_tens_a[0] + center_width[0],
            ] = tens[
                center_tens_b[0] - center_width[0]:center_tens_b[0] + center_width[0],
            ]
        else:
            raise Exception()
        return tens_new
        
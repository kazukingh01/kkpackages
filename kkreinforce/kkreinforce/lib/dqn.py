import copy
from typing import List
import torch
from torch import nn
import pandas as pd
import numpy as np
from collections import namedtuple
import random

# local package
from kkimagemods.util.logger import set_logger, set_loglevel
_logname = __name__
logger = set_logger(__name__)

Transition = namedtuple('Transition',('state', 'action', 'reward', 'state_next'))
Layer      = namedtuple("Layer", ("name", "module", "node", "params", "kwards"))


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
        for layer in layers:
            if layer.node is None:
                self.add_module(layer.name, layer.module(*layer.params, **layer.kwards))
            else:
                self.add_module(layer.name, layer.module(in_size, layer.node, *layer.params, **layer.kwards))
                in_size = layer.node
        
        # 計算後の output でどこを使うかの index を持っておく
        self.indexes = []
        self.__compile()
    

    def __compile(self):
        """
        LSTMなどは使うoutputの内容が違ってくる.
        forward 内にできるだけ if 文などの条件分岐で計算を遅くしないため
        予め計算過程を最適化しておく
        """
        for i, x in enumerate(self.modules()):
            if i == 0:
                self.indexes.append(None)
                continue # 0 番目は全体のLayerが呼び出されるため
            if type(x) == torch.nn.LSTM:
                self.indexes.append("last")
            else:
                self.indexes.append(None)


    def __call__(self, input: torch.Tensor):
        output = input.clone()
        for i, module in enumerate(self.modules()):
            if i == 0: continue
            if   self.indexes[i] is None:
                output = module(output)
            elif self.indexes[i] == "last":
                output = module(output)
                output = output[0][:, -1, :]
            else:
                output = module(output)
        return output


class DQN(object):
    def __init__(self, torch_nn: TorchNN, state_list: List[object], action_list: List[object], alpha: float, gamma: float):
        super().__init__()
        self.q     = pd.DataFrame(np.nan, index=state_list, columns=action_list, dtype=float)
        self.ndf   = self.q.values # 遅いので参照形式にしておく
        self.alpha = alpha
        self.gamma = gamma
        # index を参照するのを高速化する辞書を予め用意しておく
        self.dict_state   = {x:i for i, x in enumerate(state_list)}
        self.dict_action  = {x:i for i, x in enumerate(action_list)}
        self.index_state  = {self.dict_state[ x]:x for x in self.dict_state. keys()} # index からの逆引きも定義しておく 
        self.index_action = {self.dict_action[x]:x for x in self.dict_action.keys()} # index からの逆引きも定義しておく 

        # NN config
        ## Q Network
        self.qnet = torch_nn
        ## optimizer
        self.optimizer = torch.optim.SGD(self.qnet.parameters(), lr=0.001, momentum=0.9, weight_decay=0)
        ## loss function
        self.criterion = nn.SmoothL1Loss(reduction="sum")
        ## Freezing the target network
        self.qnet_freez = copy.deepcopy(self.qnet)
        self.qnet_freez.load_state_dict(self.qnet.state_dict().copy())
        self.synchronize_cnt = 0
        self.synchronize_max = 100
        
        # Train config
        self.batch_size = 128
        # Replay Memory
        self.memory = ReplayMemory(1000)


    def to_cuda(self):
        logger.info("START")
        # cudaへのtoは return で変数を上書きしなくても大丈夫
        # moduleがcudaにのっているかどうかは、next(model.parameters()).is_cuda で確認できる
        self.qnet.to(torch.device("cuda:0"))
        self.criterion.to(torch.device("cuda:0"))
        logger.info("END")


    def conv_onehot(self, value, calc_state=True):
        n_class, index = None, None
        if type(value) == str or type(value) == int: value = [value]
        if calc_state:
            n_class = len(self.dict_state)
            index   = [self.dict_state[x] for x in value]
        else:
            n_class = len(self.dict_action)
            index   = [self.dict_action[x] for x in value]
        return torch.eye(n_class)[index]


    def get_value(self, state: object, action: object=None) -> float:
        """
        Index が object や tuple であることを考慮して、次のように返却する
        """
        self.qnet.eval()
        with torch.no_grad():
            tens = self.conv_onehot(state, calc_state=True)
            tens = self.qnet(tens)
            if action is None:
                return tens.detach().numpy()[0]
            else:
                return tens[:, self.dict_action[action]].detach().numpy()[0]


    def get_max(self, state: object, list_action_flg: List[bool] = None):
        self.qnet.eval()
        with torch.no_grad():
            tens    = self.conv_onehot(state, calc_state=True)
            tens    = self.qnet(tens)
            ndf     = tens.detach().numpy()[0]
            val_max = np.nanmax(ndf) if list_action_flg is None else np.nanmax(ndf[list_action_flg])
            action  = self.index_action[np.where(ndf == val_max)[0].min()]
            return val_max, action


    def update(self, state, action, reward, state_next, unit_of_learning: str=None):
        """
        Q NN の更新を行う
        Q(s, a) = Q(s, a) + alpha*(r+gamma*maxQ(s')-Q(s, a))
        
        1 episode 単位での学習を考慮する
        """
        # Store the transition in memory
        self.memory.push(state, action, reward, state_next)
        if len(self.memory) < self.batch_size: return None

        # transitions は list で 各要素に'Transition',('state', 'action', 'reward', 'state_next')が入っている
        # batch は Transition で batch.state で tuple 形式で state のみが取り出せる
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(* transitions))

        state      = batch.state
        action     = batch.action
        reward     = batch.reward
        state_next = batch.state_next
 
        self.qnet.train() # train() と eval() は Dropout があるときに区別する必要がある
        self.qnet.zero_grad() # 勾配を初期化
        
        tens_act  = self.conv_onehot(action, calc_state=False)
        tens_pred = self.conv_onehot(state, calc_state=True)
        tens_pred = self.qnet(tens_pred)
        tens_pred = tens_pred * tens_act
        with torch.no_grad():
            # Double DQN では行動は学習中のネットワークで決定する. ただ、その行動における価値については、freez NN で決める
            tens_ans  = self.conv_onehot(state_next, calc_state=True)
            tens_ans1 = self.qnet(tens_ans)
            tens_ans1 = tens_ans1.max(axis=1)[1] # 最大となるところのindex
            tens_ans2 = self.qnet_freez(tens_ans)
            tens_ans  = tens_ans2[:, tens_ans1][:, 0]
            tens_max  = torch.tensor(reward) + self.gamma * tens_ans
            tens_max  = torch.cat([tens_max.reshape(-1, 1) for i in range(tens_act.shape[1])], dim=1)
            tens_ans  = tens_max * tens_act
        loss = self.criterion(tens_pred, tens_ans)
        logger.info(f"loss: {loss}", color=["BOLD","WHITE"])
        loss.backward()
        self.optimizer.step()
        self.synchronize_cnt += 1

        if self.synchronize_max < self.synchronize_cnt:
            # あるカウントが溜まったら NN を同期する
            self.qnet_freez.load_state_dict(self.qnet.state_dict().copy())
            self.synchronize_cnt = 0


class ReplayMemory(object):
    def __init__(self, capacity, unit_memory: str=None):
        self.capacity = capacity
        self.memory   = []
        self.position = 0
        self.memory_in_episode = []
        self.unit_memory = unit_memory

    def push(self, *args, on_episode=False):
        """Saves a transition."""
        if self.unit_memory is None:
            if len(self.memory) < self.capacity:
                self.memory.append(None)
            self.memory[self.position] = Transition(*args)
            self.position = (self.position + 1) % self.capacity
        elif self.unit_memory == "episode":
            # episode 単位のまとまりで、memory に保存する
            if len(self.memory) < self.capacity:
                if on_episode:
                    self.memory_in_episode.append(Transition(*args))
                else:
                    if len(self.memory) < self.capacity:
                        self.memory.append(None)
                    self.memory[self.position] = Transition(*args)
                    self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



if __name__ == "__main__":
    """ Debug code """
    list_state  = ["aaa", "bbb", "ccc", "ddd"]
    list_action = [1, 2, 3, 4]
    dqn = DQN(list_state, list_action, alpha=1, gamma=1)

    kknn = TorchNN(100,
        Layer("fc1",   nn.Linear, 128, (), {}), 
        Layer("relu1", nn.ReLU, None,  (), {}), 
        Layer("fc2",   nn.Linear, 64,  (), {}), 
    )
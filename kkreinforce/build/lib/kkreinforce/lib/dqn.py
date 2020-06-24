import copy
from typing import List, Tuple
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



class DQN(object):
    def __init__(self, 
        torch_nn: TorchNN, 
        action_list: List[object], 
        alpha: float, gamma: float, batch_size: int, 
        capacity: int, unit_memory: str=None, lr: float=0.001
    ):
        if capacity < batch_size: raise Exception("capacity is less than batch_size. ")
        super().__init__()
        self.alpha = alpha # alpha は使わない
        self.gamma = gamma
        # action は index を参照するのを高速化する辞書を予め用意しておく
        self.dict_action  = {x:i for i, x in enumerate(action_list)}
        self.index_action = {self.dict_action[x]:x for x in self.dict_action.keys()} # index からの逆引きも定義しておく 
        self.n_class_action = len(self.dict_action) # 先に計算しておく

        # NN config
        ## Q Network
        self.qnet = torch_nn
        ## optimizer
        self.optimizer = torch.optim.RMSprop(self.qnet.parameters(), lr=lr, weight_decay=0)
        ## loss function
        self.criterion = nn.SmoothL1Loss(reduction="sum")
        ## Freezing the target network
        self.qnet_freez = copy.deepcopy(self.qnet)
        self.qnet_freez.load_state_dict(self.qnet.state_dict().copy())
        self.synchronize_cnt = 0
        self.synchronize_max = 2
        
        # Train config
        self.batch_size = batch_size
        # Replay Memory
        self.memory = ReplayMemory(capacity, unit_memory=unit_memory)
        # Cuda Flg
        self.is_cuda = False


    def to_cuda(self):
        logger.info("START")
        # cudaへのtoは return で変数を上書きしなくても大丈夫
        # moduleがcudaにのっているかどうかは、next(model.parameters()).is_cuda で確認できる
        self.qnet.to(torch.device("cuda:0"))
        self.qnet_freez.to(torch.device("cuda:0"))
        self.criterion.to(torch.device("cuda:0"))
        self.is_cuda = True
        logger.info("END")


    def val_to(self, x):
        return x.to("cuda:0") if self.is_cuda else x


    def conv_onehot(self, value):
        index = None
        if   self.memory.unit_memory is None:
            if type(value) == str or type(value) == int: value = [value]
            index = [self.dict_action[x] for x in value]
            return torch.eye(self.n_class_action)[index]
        elif self.memory.unit_memory == "episode":
            if type(value) == str or type(value) == int: value = [[value]]
            index = [[self.dict_action[x] for x in listwk] for listwk in value]
            tens = [torch.eye(self.n_class_action)[x] for x in index]
            tens = torch.cat([x.reshape(1, -1, self.n_class_action) for x in tens], axis=0)
            return tens


    def get_value(self, state: object, action: object=None) -> float:
        """
        Index が object や tuple であることを考慮して、次のように返却する
        """
        self.qnet.eval()
        with torch.no_grad():
            tens = self.qnet(state)
            if action is None:
                return tens.detach().to("cpu").numpy()[0]
            else:
                return tens[:, self.dict_action[action]].detach().to("cpu").numpy()[0]


    def get_max(self, state: np.ndarray, prob_actions: np.ndarray = None):
        self.qnet.eval()
        with torch.no_grad():
            state = torch.from_numpy(state.astype(np.float32).reshape(1, *state.shape)) # 次元を1つ上げる
            state = self.val_to(state)
            tens  = self.qnet(state, option=("last" if self.memory.unit_memory == "episode" else None))
            ndf   = tens.detach().to("cpu").numpy()[-1]
            if prob_actions is not None:
                prob_actions_wk = prob_actions.copy().astype(float)
                prob_actions_wk[prob_actions_wk == 0] = np.nan
                ndf = ndf *  prob_actions_wk
            val_max = np.nanmax(ndf)
            action  = self.index_action[np.where(ndf == val_max)[0].min()]
            return val_max, action


    def update(self, state: object, action: object, reward: object, state_next: object, prob_actions: np.ndarray=None, on_episode: bool=False):
        """
        Q NN の更新を行う
        Q(s, a) = Q(s, a) + alpha*(r+gamma*maxQ(s')-Q(s, a))
        1 episode 単位での学習を考慮する
        """
        # Store the transition in memory
        self.memory.push(state, action, reward, state_next, prob_actions, on_episode=on_episode)
        if len(self.memory) < self.batch_size: return None

        if self.memory.unit_memory == "episode" and on_episode == True: return None # episode の場合は 1episode で 1回学習させる
        # episode 単位で保存されている場合は、ndarray[[state], [state], ..] になっている
        state, action, reward, state_next, prob_actions = self.memory.sample(self.batch_size)

        self.qnet.train() # train() と eval() は Dropout があるときに区別する必要がある
        self.qnet.zero_grad() # 勾配を初期化
        tens_act  = self.conv_onehot(action)
        tens_act  = self.val_to(tens_act)
        if len(tens_act.shape) == 1: tens_act = tens_act.reshape(-1, tens_act.shape[-1]) # 1次元は2次元に
        tens_pred = torch.from_numpy(state.astype(np.float32))
        tens_pred = self.val_to(tens_pred)
        tens_pred = self.qnet(tens_pred)
        tens_pred = tens_pred * tens_act.reshape(-1, tens_pred.shape[-1]) # 3次元になることは想定しない
        with torch.no_grad():
            # Double DQN では行動は学習中のネットワークで決定する. ただ、その行動における価値については、freez NN で決める
            tens_ans  = torch.from_numpy(state_next.astype(np.float32))
            tens_ans  = self.val_to(tens_ans)
            tens_ans1 = self.qnet(tens_ans, option=("not_first" if self.memory.unit_memory == "episode" else None))
            if (prob_actions == None).sum() == 0:
                prob_actions_wk = prob_actions.copy().astype(np.float32)
                prob_actions_wk[prob_actions_wk == 0] = np.nan
                prob_actions_wk = torch.from_numpy(prob_actions_wk).reshape(-1, tens_ans1.shape[-1])
                prob_actions_wk  = self.val_to(prob_actions_wk)
                tens_ans1 = tens_ans1 * prob_actions_wk # 3次元になることは想定しない
                tens_ans1[torch.isnan(tens_ans1)] = float("-inf") # nan だと max で nan になるので -inf を埋める
            tens_ans0, tens_ans1 = tens_ans1.max(axis=1) # 最大となるところのindex
            tens_ans2 = self.qnet_freez(tens_ans, option=("not_first" if self.memory.unit_memory == "episode" else None))
            tens_ans2[torch.isinf(tens_ans0), :] = 0
            tens_ans  = tens_ans2[:, tens_ans1][torch.eye(tens_ans1.shape[0]).bool()]
            tens_rwd  = torch.tensor(reward.astype(np.float32)).reshape(-1)
            tens_rwd  = self.val_to(tens_rwd)
            tens_max  = tens_rwd + self.gamma * tens_ans
            tens_max  = torch.cat([tens_max.reshape(-1, 1) for i in range(tens_act.shape[-1])], dim=1)
            tens_ans  = tens_max * tens_act.reshape(-1, tens_act.shape[-1]) # 3次元になることは想定しない
        loss = self.criterion(tens_pred, tens_ans)
        logger.info(f"loss: {loss}", color=["BOLD","WHITE"])
        loss.backward()
        self.optimizer.step()
        self.synchronize_cnt += 1

        if self.synchronize_cnt % self.synchronize_max == self.synchronize_max - 1:
            # あるカウントが溜まったら NN を同期する
            self.qnet_freez.load_state_dict(self.qnet.state_dict().copy())
            self.synchronize_cnt = 0



Transition = namedtuple('Transition',('state', 'action', 'reward', 'state_next', 'prob_actions'))
class ReplayMemory(object):

    def __init__(self, capacity: int, unit_memory: str=None):
        super(ReplayMemory, self).__init__()
        self.capacity = capacity + 1 # best episode 用に追加
        self.memory   = []
        self.position = 0
        self.memory_in_episode = []
        self.unit_memory = unit_memory
        self.reward_max  = -np.inf
        self.trans_best  = None

    def push(self, *args, on_episode=False):
        """
        Saves a transition. consider episode. 
        """
        if self.unit_memory is None:
            if len(self.memory) < self.capacity:
                self.memory.append(None)
            self.memory[self.position] = copy.deepcopy(Transition(*args))
            self.position = (self.position + 1) % self.capacity
        elif self.unit_memory == "episode":
            #  episode 単位のまとまりで、memory に保存する
            self.memory_in_episode.append(copy.deepcopy(Transition(*args)))
            if on_episode == False:
                if len(self.memory) < self.capacity:
                    self.memory.append(None)
                # Best episode は 消さないようにしたい
                if self.reward_max <= self.memory_in_episode[-1].reward:
                    self.reward_max = self.memory_in_episode[-1].reward
                    self.memory[0]  = self.memory_in_episode.copy()
                self.memory[self.position] = self.memory_in_episode.copy()
                self.position = (self.position + 1) % self.capacity
                if self.position == 0: self.position += 1 # 0 index は特別なのでずらす
                self.memory_in_episode = []
        else:
            raise Exception(f"We don't consider this unit: {self.unit_memory}")

    def sample(self, batch_size: int=1, indexes: List[int] = None) -> (List[object], List[object], List[object], List[object], List[object]):
        # indexes が指定されたらそっちを優先する
        transitions = None
        if indexes is None:
            transitions = random.sample(self.memory, batch_size)
        else:
            transitions = []
            for x in indexes: transitions.append(self.memory[x])
        if self.unit_memory is None:
            batch = Transition(*zip(* transitions))
            return np.array(batch.state), np.array(batch.action), np.array(batch.reward), np.array(batch.state_next), np.array(batch.prob_actions)
        elif self.unit_memory == "episode":
            # best episode を強制追加
            #transitions.append(self.memory[0])
            state        = np.array([[x.state        for x in episode] for episode in transitions])
            action       = np.array([[x.action       for x in episode] for episode in transitions])
            reward       = np.array([[x.reward       for x in episode] for episode in transitions])
            state_next   = np.array([[x.state_next   for x in episode] for episode in transitions])
            prob_actions = np.array([[x.prob_actions for x in episode] for episode in transitions])
            ## LSTMで扱えるようにepisodeの順番を記憶しているが、state_nextには最初の状態が抜けているため、そこを補間してやる
            state_next   = np.insert(state_next, 0, state[::,0].copy(), axis=1)
            return  state, action, reward, state_next, prob_actions
        else:
            raise Exception(f"We don't consider this unit: {self.unit_memory}")

    def __len__(self):
        return len(self.memory)


if __name__ == "__main__":
    """ Debug code """
    list_state  = ["aaa", "bbb", "ccc", "ddd"]
    list_action = [1, 2, 3, 4]
    torch_nn = TorchNN(
        len(list_action),
        Layer("lstm",  torch.nn.LSTM,   128,   "rnn_all", (), {}),
        Layer("relu1", torch.nn.ReLU,   None,  None, (), {}),
        Layer("fc2",   torch.nn.Linear, 128,   None, (), {}),
        Layer("relu2", torch.nn.ReLU,   None,  None, (), {}),
        Layer("fc3",   torch.nn.Linear, len(list_action), None, (), {}),
    )
    qtable = DQN(torch_nn, list_state, list_action, alpha=0.5, gamma=0.5, batch_size=128, capacity=200, unit_memory="episode")
    for i in range(100):
        x = np.random.permutation(np.arange(len(list_state)))
        y = np.random.permutation(np.arange(len(list_action)))
        qtable.update(list_state[x[0]], list_action[y[0]], 1, list_state[x[1]], on_episode=True)
        if i % 10 == 0:
            qtable.update(list_state[x[0]], list_action[y[0]], 1, list_state[x[1]], on_episode=False)
    state, action, reward, state_next = qtable.memory.sample(4)
    state = qtable.conv_onehot(state)
    
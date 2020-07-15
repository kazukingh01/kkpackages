import copy, random
import numpy as np
import pandas as pd
import torch
from typing import List, Tuple
from collections import namedtuple

# local package
from kkreinforce.lib.kknn import TorchNN
from kkimagemods.util.logger import set_logger, set_loglevel
logger = set_logger(__name__)


class RLBase(object):
    """
    強化学習における共通的な枠組みを定義する.
    ※本的にこのBaseクラスを継承し、必要なfunctionを子クラスで実装する.
    state(t-1), action(t-1), reward(t), state(t)
    全ての遷移は transition function で行われるものとし、それ以外のfuncでは
    self.state_now = xxx などは実装しないようにする
    """
    def __init__(self):
        super(RLBase, self).__init__()
        # state, action, reward
        self.state_prev: object  = None
        self.action_prev: object = None
        self.reward_now: object  = None
        self.state_now: object   = None
        # others
        self.step: int           = 0
        self.episode: int        = 0
        self.is_eval: bool       = None

    def initialize(self):
        """
        state_prev, action_prev, reward_now, state_now を初期化する
        """
        self.state_prev:  object
        self.action_prev: object
        self.reward_now:  object
        self.state_now:   object
        raise NotImplementedError()

    def init(self):
        logger.info("Initialize", color=["BOLD", "GREEN"])
        self.is_eval = False
        self.step    = 0
        self.initialize()

    def state(self, state_prev: object=None, action_prev: object=None) -> object:
        """
        state_prevとaction_prevから状態遷移確率に従い、次のstateを決定する
        """
        state_prev  = self.state_prev  if state_prev  is None else state_prev 
        action_prev = self.action_prev if action_prev is None else action_prev
        raise NotImplementedError()
        return object()

    def action(self, state_now: object=None) -> object:
        """
        state_now から policy に従って決定されるactionを返却する
        """
        state_now = self.state_now if state_now is None else state_now
        raise NotImplementedError()
        return object()

    def reward(self, state_prev: object=None, action_prev: object=None, state_now: object=None) -> object:
        """
        state_prev と action_prev (と state_now) から決まる reward を返却する
        """
        state_prev  = self.state_prev  if state_prev  is None else state_prev 
        action_prev = self.action_prev if action_prev is None else action_prev
        state_now   = self.state_now   if state_now   is None else state_now
        raise NotImplementedError()
        return object()

    def transition_before_all(  self): pass
    def transition_before_state(self): pass
    def transition_after_state( self): pass
    def transition_after_all(   self): pass

    def transition(self, action: object = None, next_state: object = None):
        """
        state(t), action(t), reward(t+1), state(t+1)の遷移を行う
        action(t), state(t+1) が確率的な場合を考慮し、引数に入力された場合は
        その行動や状態を確定的にし、Noneの場合は action(), state() に従う事とする
        """
        self.transition_before_all()
        self.action_prev = action if action is not None else self.action() # state_now によって決定されるため一番先にactionを決定
        self.state_prev  = self.state_now
        self.transition_before_state()
        self.state_now   = next_state if next_state is not None else self.state()
        self.transition_after_state()
        self.reward_now = self.reward()
        self.step += 1 # step を加算する
        self.transition_after_all()

    def is_finish(self)  -> int:
        """
        episode の終了判定を実装する
        """
        raise NotImplementedError()

    def train_before_episode(self): pass
    def train_before_step   (self): pass
    def train_after_step    (self): pass
    def train_after_episode (self): pass

    def train(self, n_episode: int=100):
        """
        学習する.
        ※続けて学習する可能性もあるので、self.episode は初期化しない
        """
        self.is_eval = False
        for _ in range(n_episode):
            logger.info(f"episode: {self.episode}", color=["BOLD", "BLUE"])
            self.init()
            self.train_before_episode()
            while self.is_finish() == 0:
                self.train_before_step()
                self.transition()
                self.train_after_step()
            self.train_after_episode()
            self.episode += 1



class StateManager(object):
    """
    状態管理のためのclass
    状態の入力:str -> onehot にするなど、NNの入力に適した形に変換する
    """
    def __init__(self):
        super(StateManager, self).__init__()
        self.dict_state = {}
        self.list_names = []
    
    def __len__(self):
        count = 0
        for x in self.list_names:
            if   self.dict_state[x]["type"] == "onehot":
                count += self.dict_state[x]["nclass"]
            elif self.dict_state[x]["type"] == "onehot_cntup":
                count += self.dict_state[x]["nclass"]
            elif self.dict_state[x]["type"] == "onehot_binary":
                count += self.dict_state[x]["nclass"]
            elif self.dict_state[x]["type"] == "numeric_bins":
                count += self.dict_state[x]["nclass"]
            else:
                count += 1
        return count
    
    def set_state(self, name: str, state_type: str, state_list: List[object] = None, options: dict = {}):
        """
        Params::
            name: state の名称
            state_type: numeric, list, binary, onehot, onehot_cntup, onehot_binary
                ※ binary の場合、その値が入ってこなければ 0 を入力する
            state_list: str, onehot の場合は必要
        """
        self.list_names.append(name)
        state = {}
        state["type"]   = state_type
        if state_type == "numeric_bins":
            bins    = np.linspace(options["bin_min"], options["bin_max"], options["bins"])
            state["state"]  = bins
            state["index"]  = lambda x: np.digitize(x, bins)
            state["nclass"] = len(bins) + 1
        else:
            state["state"]  = {x:i for i, x in enumerate(state_list)} if state_list is not None else {}
            state["index"]  = {i:x for i, x in enumerate(state_list)} if state_list is not None else {}
            state["nclass"] = len(state_list) if state_list is not None else 0
        state["value"]  = False if state_type == "binary" else None
        self.dict_state[name] = state
    
    def pattern(self) -> List[Tuple[object]]:
        """
        Return::
            考えられる組み合わせの状態を全て返却する. numeric や onehot の場合は error とする
        """
        listwk = []
        for x in self.list_names:
            if   self.dict_state[x]["type"] == "list":
                listwk.append(list(self.dict_state[x]["state"].values()))
            elif self.dict_state[x]["type"] == "binary":
                listwk.append([0, 1])
            elif self.dict_state[x]["type"] == "numeric_bins":
                listwk.append(np.arange(0, self.dict_state[x]["nclass"]).tolist())
            elif self.dict_state[x]["type"] == "onehot":
                raise Exception(f'We can not calculate {self.dict_state[x]["type"]} type.')
            elif self.dict_state[x]["type"] == "numeric":
                raise Exception(f'We can not calculate {self.dict_state[x]["type"]} type.')
            elif self.dict_state[x]["type"] == "onehot_cntup":
                raise Exception(f'We can not calculate {self.dict_state[x]["type"]} type.')
            elif self.dict_state[x]["type"] == "onehot_binary":
                raise Exception(f'We can not calculate {self.dict_state[x]["type"]} type.')
        df = None
        for i, listwkwk in enumerate(listwk):
            if df is None:
                df = pd.DataFrame(listwkwk, columns=[i])
                df["__work"] = 1
            else:
                dfwk = pd.DataFrame(listwkwk, columns=[i])
                dfwk["__work"] = 1
                df = pd.merge(df, dfwk, how="left", on="__work")
        df = df.drop(columns=["__work"])
        return df.apply(lambda x: tuple(x),axis=1).to_list()
    
    def conv(self) -> np.ndarray:
        output = np.zeros(0).astype(float)
        for x in self.list_names:
            output = np.append(output, self.dict_state[x]["value"])
        return output.reshape(-1).astype(int).copy()
    
    def conv_tmp(self, values: dict) -> np.ndarray:
        """
        set_value -> conv では状態が変わってしまうため、set_value せずに conv の結果を出せる関数を用意する
        """
        dict_state = copy.deepcopy(self.dict_state)
        self.set_values(values)
        output = self.conv()
        self.dict_state = dict_state # 元に戻す
        return output

    def set_value(self, name: str, value: object):
        if   self.dict_state[name]["type"] == "numeric":
            self.dict_state[name]["value"] = value
        elif self.dict_state[name]["type"] == "numeric_bins":
            self.dict_state[name]["value"] = self.dict_state[name]["index"](value) # callable function.(lambda x)
        elif self.dict_state[name]["type"] == "list":
            self.dict_state[name]["value"] = self.dict_state[name]["state"][value]
        elif self.dict_state[name]["type"] == "onehot":
            val = np.identity(self.dict_state[name]["nclass"])[self.dict_state[name]["state"][value]]
            self.dict_state[name]["value"] = val
        elif self.dict_state[name]["type"] == "binary":
            self.dict_state[name]["value"] = True
        elif self.dict_state[name]["type"] == "onehot_binary":
            if self.dict_state[name]["value"] is None:
                self.dict_state[name]["value"] = np.zeros(self.dict_state[name]["nclass"]).astype(bool)
            ndf = self.dict_state[name]["value"] # 参照形式で修正
            ndf[self.dict_state[name]["state"][value]] = True
        elif self.dict_state[name]["type"] == "onehot_cntup":
            if self.dict_state[name]["value"] is None:
                self.dict_state[name]["value"] = np.zeros(self.dict_state[name]["nclass"])
            ndf = self.dict_state[name]["value"] # 参照形式で修正
            ndf[self.dict_state[name]["state"][value]] += 1
    
    def set_values(self, values: dict):
        for x, y in values.items():
            self.set_value(x, y)

    def reset_value(self):
        for x in self.list_names:
            self.dict_state[x]["value"] = False if self.dict_state[x]["type"] == "binary" else None



Transition = namedtuple('Transition',('state', 'action', 'reward', 'state_next', 'prob_actions', 'on_episode'))
class ReplayMemory(object):

    def __init__(self, capacity: int, unit_memory: str=None, memory_best: bool=False):
        super(ReplayMemory, self).__init__()
        self.capacity = capacity + 1 # best episode 用に追加
        self.memory   = []
        self.position = 0
        self.memory_in_episode = []
        self.unit_memory = unit_memory
        self.memory_best = memory_best
        self.reward_max  = -np.inf
        self.trans_best  = None

    def push(self, *args, on_episode=False):
        """
        Saves a transition. consider episode.
        """
        if self.unit_memory is None:
            if len(self.memory) < self.capacity:
                self.memory.append(None)
            self.memory[self.position] = copy.deepcopy(Transition(*args, on_episode))
            self.position = self.position + 1
            if self.capacity != float("inf"):
                self.position = self.position % self.capacity
        elif self.unit_memory == "episode":
            #  episode 単位のまとまりで、memory に保存する
            self.memory_in_episode.append(copy.deepcopy(Transition(*args, on_episode)))
            if on_episode == False:
                if len(self.memory) < self.capacity:
                    self.memory.append(None)
                # Best episode は 消さないようにしたい
                if self.memory_best and self.reward_max <= self.memory_in_episode[-1].reward:
                    self.reward_max = self.memory_in_episode[-1].reward
                    self.memory[0]  = self.memory_in_episode.copy()
                self.memory[self.position] = self.memory_in_episode.copy()
                self.position = self.position + 1
                if self.capacity != float("inf"):
                    self.position = self.position % self.capacity
                if self.memory_best and self.position == 0: self.position += 1 # 0 index は特別なのでずらす
                self.memory_in_episode = []
        else:
            raise Exception(f"We don't consider this unit: {self.unit_memory}")

    def sample(self, batch_size: int=1, indexes: List[int] = None) -> (List[object], List[object], List[object], List[object], List[object]):
        # indexes が指定されたらそっちを優先する
        transitions = None
        if indexes is None:
            if batch_size < 0 or batch_size > len(self.memory):
                transitions = random.sample(self.memory, len(self.memory))
            else:
                transitions = random.sample(self.memory, batch_size)
        else:
            transitions = []
            for x in indexes: transitions.append(self.memory[x])
        if self.unit_memory is None:
            batch = Transition(*zip(* transitions))
            return np.array(batch.state), np.array(batch.action), np.array(batch.reward), np.array(batch.state_next), np.array(batch.prob_actions), np.array(batch.on_episode)
        elif self.unit_memory == "episode":
            # best episode を強制追加
            #transitions.append(self.memory[0])
            state        = np.array([[x.state        for x in episode] for episode in transitions])
            action       = np.array([[x.action       for x in episode] for episode in transitions])
            reward       = np.array([[x.reward       for x in episode] for episode in transitions])
            state_next   = np.array([[x.state_next   for x in episode] for episode in transitions])
            prob_actions = np.array([[x.prob_actions for x in episode] for episode in transitions])
            on_episode   = np.array([[x.on_episode   for x in episode] for episode in transitions])
            ## LSTMで扱えるようにepisodeの順番を記憶しているが、state_nextには最初の状態が抜けているため、そこを補間してやる
            state_next   = np.insert(state_next, 0, state[::,0].copy(), axis=1)
            return  state, action, reward, state_next, prob_actions, on_episode
        else:
            raise Exception(f"We don't consider this unit: {self.unit_memory}")

    def reset(self):
        self.memory   = []
        self.position = 0

    def __len__(self):
        return len(self.memory)



class ActionValueFunction(object):
    def get_value(self): raise NotImplementedError()
    def get_max  (self): raise NotImplementedError()
    def update   (self): raise NotImplementedError()
    def store    (self): pass



class RLBaseNN(ActionValueFunction):
    """
    強化学習で使用するNNのBaseクラス
    NNの最終ノードはactionに関連する価値や確率が出力されるものとする
    """
    def __init__(self, 
        torch_nn: TorchNN, action_list: List[object], 
        batch_size: int, capacity: int, unit_memory: str=None, 
        lr: float=0.001
    ):
        if capacity < batch_size:
            raise Exception("capacity is less than batch_size. ")
        super().__init__()
        # action は index を参照するのを高速化する辞書を予め用意しておく
        self.dict_action  = {x:i for i, x in enumerate(action_list)}
        self.index_action = {self.dict_action[x]:x for x in self.dict_action.keys()} # index からの逆引きも定義しておく 
        self.n_class_action = len(self.dict_action) # 先に計算しておく
        # NN config
        self.nn = torch_nn
        ## optimizer
        self.optimizer = torch.optim.RMSprop(self.nn.parameters(), lr=lr, weight_decay=0)
        # Train config
        self.batch_size = batch_size
        # Replay Memory
        self.memory = ReplayMemory(capacity, unit_memory=unit_memory)
        # Cuda Flg
        self.is_cuda = False

    def to_cuda(self):
        """ GPUの使用をONにする """
        self.nn.to(torch.device("cuda:0")) # moduleがcudaにのっているかどうかは、next(model.parameters()).is_cuda で確認できる
        self.is_cuda = True

    def val_to(self, x):
        """ GPUの使用がONであればGPUに乗せる """
        return x.to("cuda:0") if self.is_cuda else x

    def conv_onehot(self, action: object) -> torch.Tensor:
        """
        action の object を NN の最終ノードの数で OneOot に変換する
        Return::
            "test" という input action に対して、[0, 1, 0, 0, ..., 0] のような変換する
        Params::
            action: str or List[str]
        """
        index = None
        if   self.memory.unit_memory is None:
            if type(action) == str or type(action) == int: action = [action]
            index = [self.dict_action[x] for x in action]
            return torch.eye(self.n_class_action)[index]
        elif self.memory.unit_memory == "episode":
            if type(action) == str or type(action) == int: action = [[action]]
            index = [[self.dict_action[x] for x in listwk] for listwk in action]
            tens = [torch.eye(self.n_class_action)[x] for x in index]
            tens = torch.cat([x.reshape(1, -1, self.n_class_action) for x in tens], axis=0)
            return tens

    def get_value(self, values: np.ndarray, action: object=None) -> np.ndarray:
        """
        NN に values を入力して、そのoutput を返却する.
        Return::
            action を指定した場合は、float[actionノードの出力]
            Noneの場合は output ノードを numpy 形式で返却する
        """
        self.nn.eval()
        with torch.no_grad():
            values = torch.from_numpy(values.astype(np.float32).reshape(1, *values.shape)) # 次元を1つ上げる
            values = self.val_to(values)
            tens  = self.nn(values, option=("last" if self.memory.unit_memory == "episode" else None))
            logger.debug(f"values: {tens}", color=["BOLD"])
            if action is None:
                return tens.detach().to("cpu").numpy()[-1]
            else:
                return tens[:, self.dict_action[action]].detach().to("cpu").numpy()[-1]

    def get_max(self, values: np.ndarray, prob_actions: np.ndarray = None) -> (float, object, ):
        """
        NN に values を入力して、そのoutput のうち最大の行動となる value と action を返却する.
        Params::
            values: NN の input
            prob_actions: 出力ノードのうち max の計算で無視するノードをbooleanで指定する
        """
        self.nn.eval()
        with torch.no_grad():
            ndf = self.get_value(values=values, action=None)
            if prob_actions is not None:
                prob_actions_wk = prob_actions.copy().astype(float)
                prob_actions_wk[prob_actions_wk == 0] = np.nan
                ndf = ndf *  prob_actions_wk
            val_max = np.nanmax(ndf)
            action  = self.index_action[np.where(ndf == val_max)[0].min()]
            logger.debug(f'val: {ndf}, action: {action}')
            return val_max, action
    
    def store(self, state: object, action: object, reward: object, state_next: object, prob_actions: np.ndarray=None, on_episode: bool=False):
        self.memory.push(state, action, reward, state_next, prob_actions, on_episode=on_episode)

    def update(self):
        """
        NN の back propagation
        """
        if self.batch_size > 0 and len(self.memory) < self.batch_size: return None
        # Replay Memory から sample を random に取り出す
        ## episode 単位で保存されている場合は、ndarray[[state], [state], ..] になっている
        state, action, reward, state_next, prob_actions, on_episode = self.memory.sample(self.batch_size)
        self.nn.train() # train() と eval() は Dropout があるときに区別する必要がある
        self.nn.zero_grad() # 勾配を初期化
        self.update_main(state, action, reward, state_next, prob_actions, on_episode)

    def update_main(self, state, action, reward, state_next, prob_actions, on_episode):
        """
        loss の計算を行い、loss.backward() まで実装する
        """
        raise NotImplementedError

from typing import List
import numpy as np
import torch

# local package
from kkreinforce.lib.kkrl import RLBase, ReplayMemory, Transition, RLBaseNN
from kkreinforce.lib.kknn import TorchNN
from kkimagemods.util.logger import set_logger, set_loglevel
logger = set_logger(__name__)


class PolicyGradientNN(RLBaseNN):
    def __init__(self, 
        torch_nn: TorchNN, action_list: List[object], unit_memory: str=None, lr: float=0.001, **kward
    ):
        super(PolicyGradientNN, self).__init__(
            torch_nn, action_list, batch_size=-1, capacity=float("inf"), 
            unit_memory=unit_memory, lr=lr
        )

    def update_main(self, state, action, reward, state_next, prob_actions, on_episode):
        """
        PolicyGradientNN の Loss 計算を行う
        """
        tens_act  = self.conv_onehot(action)
        tens_act  = self.val_to(tens_act)
        if len(tens_act.shape) == 1: tens_act = tens_act.reshape(-1, tens_act.shape[-1]) # 1次元は2次元に
        tens_pred = torch.from_numpy(state.astype(np.float32))
        tens_pred = self.val_to(tens_pred)
        tens_pred = self.nn(tens_pred)
        tens_pred = tens_pred[tens_act.to(bool)] # reshape(-1) は念の為しない
        with torch.no_grad():
            tens_rwd = torch.tensor(reward.astype(np.float32)).reshape(-1)
            tens_rwd = self.val_to(tens_rwd)
            tens_rwd = (tens_rwd - tens_rwd.mean()) / (tens_rwd.std() + 1e-9)
        loss      = torch.sum(-1 * torch.log(tens_pred) * tens_rwd)
        logger.info(f"loss: {loss}", color=["BOLD","WHITE"])
        loss.backward()
        self.optimizer.step()



class PolicyGradient(RLBase):
    def __init__(self, policy: PolicyGradientNN, list_action: List[str], gamma: float=0.95, capacity: int=100):
        super(PolicyGradient, self).__init__()
        self.policy = policy
        self.list_action: List[str] = list_action
        self.gamma: float = gamma
        self.memory: ReplayMemory = None # episode 単位で記憶するのに使用する

    def action(self, state_now: object=None) -> object:
        """
        state_now から policy に従って決定されるactionを返却する
        self.policy(state_now) から返却されるものは行動空間内の確率分布とする
        """
        state_now = self.state_now if state_now is None else state_now
        actions   = self.policy.get_value(state_now)
        actions   = np.cumsum(actions) # actions は1に規格化されている数値.確率と見なす
        actions[-1] += 1.0 # 下記の条件式でoutofmemoryしないように最後に1を足しておく
        index     = np.where(np.random.uniform() >= actions)[0]
        if index.shape[0] == 0:
            index = 0
        else:
            index = index[-1] + 1
        return self.list_action[index]
    
    def train_before_episode(self):
        self.memory = ReplayMemory(float("inf"))

    def train_after_step(self):
        self.memory.push(self.state_prev, self.action_prev, self.reward_now, self.state_now, None, on_episode=(self.is_finish() == 0))

    def train_after_episode(self):
        # 最新の方策で学習させる必要があるので、memory を reset する
        self.policy.memory.reset()
        # reward を計算し直して格納する
        reward = np.array(Transition(*zip(* self.memory.memory)).reward)
        for i, trans in enumerate(self.memory.memory):
            r = reward[i:].copy()
            r = (r * (self.gamma ** np.arange(r.shape[0]))).sum()
            self.policy.store(trans.state, trans.action, r, trans.state_next, trans.prob_actions, on_episode=(i < len(self.memory.memory) - 1))
        self.policy.update()

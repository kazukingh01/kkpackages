import pandas as pd
import numpy as np
import torch
from typing import List, Tuple
import copy

# local package
from kkreinforce.lib.kkrl import RLBase, StateManager, ReplayMemory, ActionValueFunction, RLBaseNN
from kkreinforce.lib.kknn import TorchNN
from kkimagemods.util.logger import set_logger, set_loglevel
logger = set_logger(__name__)



class QTable(ActionValueFunction):
    """
    Q-Learning における行動価値関数Qをテーブル形式で実装する.
    状態空間は離散値である必要がある
    """
    def __init__(self, state_list: List[object], action_list: List[object], alpha: float, gamma: float):
        super().__init__()
        # Q table
        self.q     = pd.DataFrame(np.nan, index=state_list, columns=action_list, dtype=float)
        self.ndf   = self.q.values # 遅いので参照形式にしておく
        # Q-learning のハイパーパラメータ
        self.alpha = alpha
        self.gamma = gamma
        # index を参照するのを高速化する辞書を予め用意しておく
        self.dict_state  = {x:i for i, x in enumerate(self.q.index  )}
        self.dict_action = {x:i for i, x in enumerate(self.q.columns)}


    def get_value(self, state: object, action: object=None) -> float:
        """
        Index が object や tuple であることを考慮して、次のように返却する
        Return::
        """
        if action is None:
            # state のみを想定
            ndf = self.ndf[self.dict_state[state], :].reshape(-1).copy()
            ndf[ndf == np.nan] = 0.0
            return ndf
        else:
            val = self.ndf[self.dict_state[state], self.dict_action[action]]
            return 0.0 if np.isnan(val) else val


    def get_max(self, state: object, prob_actions: np.ndarray = None):
        """
        prob_actions は action の遷移確率を表す.
        """
        se_qvalue = self.q.iloc[self.dict_state[state], :].copy()
        if prob_actions is not None:
            prob_actions_wk = prob_actions.astype(float).copy()
            prob_actions_wk[prob_actions_wk == 0] = np.nan 
            se_qvalue = se_qvalue * prob_actions_wk
        action = se_qvalue.idxmax()
        if type(action) == np.float and np.isnan(action):
            if prob_actions is None:
                action = self.q.columns[0]
            else:
                action = self.q.columns[prob_actions > 0]
                if action.shape[0] > 0: action = action[0]
                else: action = None
        val = se_qvalue.max()
        val = 0 if np.isnan(val) else val
        logger.debug(f'state: {state}, prob_actions: {prob_actions}, val: {val}, action: {action}')
        return val, action


    def update(self, state, action, reward, state_next, prob_actions: np.ndarray=None, **kward):
        """
        Q table の更新を行う
        Q(s, a) = Q(s, a) + alpha*(r+gamma*maxQ(s')-Q(s, a))
        """
        q     = self.get_value(state, action=action) # Q(s, a)
        max_q = self.get_max(state_next, prob_actions=prob_actions)[0]
        q_update = (self.alpha * (reward + (self.gamma * max_q) - q))
        logger.debug(f'q now: {q}, max q: {max_q}, reward: {reward}, update q: {q_update}')

        # 参照形式で更新する
        self.ndf[self.dict_state[state], self.dict_action[action]] = q + q_update



class DQN(RLBaseNN):
    """
    Q-Learning における行動価値関数QをNNで実装する.
    """
    def __init__(self, 
        torch_nn: TorchNN, action_list: List[object], 
        gamma: float, 
        batch_size: int, capacity: int, unit_memory: str=None, lr: float=0.001, **kward
    ):
        super(DQN, self).__init__(
            torch_nn, action_list, batch_size, capacity, 
            unit_memory=unit_memory, lr=lr
        )
        self.gamma = gamma
        # loss function
        self.criterion = torch.nn.SmoothL1Loss(reduction="sum")
        # Freezing the target network
        self.nn_freez = copy.deepcopy(self.nn)
        self.nn_freez.load_state_dict(self.nn.state_dict().copy())
        self.synchronize_cnt = 0
        self.synchronize_max = 2


    def to_cuda(self):
        super().to_cuda()
        self.nn_freez.to(torch.device("cuda:0"))
        self.criterion.to(torch.device("cuda:0"))


    def update_main(self, state, action, reward, state_next, prob_actions):
        """
        DQN の Loss 計算を行う
        """
        tens_act  = self.conv_onehot(action)
        tens_act  = self.val_to(tens_act)
        if len(tens_act.shape) == 1: tens_act = tens_act.reshape(-1, tens_act.shape[-1]) # 1次元は2次元に
        tens_pred = torch.from_numpy(state.astype(np.float32))
        tens_pred = self.val_to(tens_pred)
        tens_pred = self.nn(tens_pred)
        tens_pred = tens_pred * tens_act.reshape(-1, tens_pred.shape[-1]) # 3次元になることは想定しない
        with torch.no_grad():
            # Double DQN では行動は学習中のネットワークで決定する. ただ、その行動における価値については、freez NN で決める
            tens_ans  = torch.from_numpy(state_next.astype(np.float32))
            tens_ans  = self.val_to(tens_ans)
            tens_ans1 = self.nn(tens_ans, option=("not_first" if self.memory.unit_memory == "episode" else None))
            if (prob_actions == None).sum() == 0:
                prob_actions_wk = prob_actions.copy().astype(np.float32)
                prob_actions_wk[prob_actions_wk == 0] = np.nan
                prob_actions_wk = torch.from_numpy(prob_actions_wk).reshape(-1, tens_ans1.shape[-1])
                prob_actions_wk  = self.val_to(prob_actions_wk)
                tens_ans1 = tens_ans1 * prob_actions_wk # 3次元になることは想定しない
                tens_ans1[torch.isnan(tens_ans1)] = float("-inf") # nan だと max で nan になるので -inf を埋める
            tens_ans0, tens_ans1 = tens_ans1.max(axis=1) # 最大となるところのindex
            tens_ans2 = self.nn_freez(tens_ans, option=("not_first" if self.memory.unit_memory == "episode" else None))
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
            self.nn_freez.load_state_dict(self.nn.state_dict().copy())
            self.synchronize_cnt = 0



class QLearn(RLBase):
    """
    Q-Learning. 継承して使用する
    """
    def __init__(self, qfunc: ActionValueFunction, epsilon: float):
        """
        Params::
            qtable: QTable or DQN
            epsilon: ランダム行動を取る確率
        """
        super(QLearn, self).__init__()
        self.prob_actions: np.ndarray[bool] = None # 行動を制御するためのフラグの導入
        self.qfunc   = qfunc   # Q関数
        self.epsilon = epsilon # random行動確率
    
    def action_best(self) -> object:
        """
        state_now から決定される最適 action を返却する
        """
        raise NotImplementedError()

    def action_random(self) -> object:
        """
        state_now から決定される random action を返却する
        """
        raise NotImplementedError()

    def action(self, state_now: object=None) -> object:
        # greedy選択
        if np.random.uniform() < self.epsilon:  # random行動
            action = self.action_random()
        else:
            action = self.action_best()
        logger.debug(f'action: {action}.')
        return action

    def q_update(self):
        """
        Q値の更新を行う
        Q(s, a) = Q(s, a) + alpha*(r+gamma*maxQ(s')-Q(s, a))
        #self.qfunc.update(self.state_prev, self.action_prev, self.reward_now, self.state_now, prob_actions=self.prob_actions, on_episode=(self.is_finish() == False))
        """
        logger.debug(f"state_prev: {self.state_prev}, action_prev: {self.action_prev}, reward_now: {self.reward_now}, state_now: {self.state_now}", color=["YELLOW"])
        raise NotImplementedError()
    
    def train_after_step(self):
        self.q_update()

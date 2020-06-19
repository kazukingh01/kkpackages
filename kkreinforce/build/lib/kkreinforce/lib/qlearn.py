import pandas as pd
import numpy as np
from typing import List, Tuple

# local package
from kkimagemods.util.logger import set_logger, set_loglevel
logger = set_logger(__name__)


class QTable(object):
    def __init__(self, state_list: List[object], action_list: List[object], alpha: float, gamma: float):
        super().__init__()
        self.q     = pd.DataFrame(np.nan, index=state_list, columns=action_list, dtype=float)
        self.ndf   = self.q.values # 遅いので参照形式にしておく
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


    def update(self, state, action, reward, state_next, prob_actions: np.ndarray=None, on_episode: bool=False):
        """
        Q table の更新を行う
        on_episode は使わないが、DQNで必要なので定義しておく
        """
        # Q(s, a) = Q(s, a) + alpha*(r+gamma*maxQ(s')-Q(s, a))
        q     = self.get_value(state, action=action) # Q(s, a)
        max_q = self.get_max(state_next, prob_actions=prob_actions)[0]
        q_update = (self.alpha * (reward + (self.gamma * max_q) - q))
        logger.debug(f'q now: {q}, max q: {max_q}, reward: {reward}, update q: {q_update}')

        # 参照形式で更新する
        self.ndf[self.dict_state[state], self.dict_action[action]] = q + q_update



class QLearn(object):
    """
    仮想クラスっぽく作る. overrideする関数は必須
    """
    def __init__(self, qtable: QTable, epsilon: float):
        """
        Params::
            qtable: QTable or DQN
            epsilon: ランダム行動を取る確率
        """
        self.qtable       = qtable
        self.state_now    = None
        self.state_prev   = None
        self.action_prev  = None
        self.reward_prev  = None
        self.prob_actions = None
        self.step         = 0
        self.episode      = 0
        self.epsilon      = epsilon
        self.is_eval      = False # 学習外ではこのフラグで状態の更新などを制御する
    
    def initialize(self):
        """
        state_prev, action_prev, reward_prev, state_now, self.action_prev を初期化する
        """
        raise NotImplementedError()

    def init(self):
        """ オーバーライドしない """
        logger.info("Initialize", color=["BOLD", "GREEN"])
        self.is_eval = False
        self.step = 0
        self.initialize()

    def state(self, state_prev: object=None, action_prev: object=None) -> object:
        """
        state_prev と action_prev から決まる state を返却する
        """
        state_prev  = self.state_prev  if state_prev  is None else state_prev 
        action_prev = self.action_prev if action_prev is None else action_prev 
        raise NotImplementedError()
        return object()

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

    def reward(self, state_prev: object=None, action_prev: object=None) -> object:
        """
        state_prev と action_prev から決まる reward を返却する
        """
        state_prev  = self.state_prev  if state_prev  is None else state_prev 
        action_prev = self.action_prev if action_prev is None else action_prev 
        raise NotImplementedError()
        return object()

    def transition_before(self): pass
    def transition_middle(self): pass
    def transition_after( self): pass

    def transition(self, action: object = None):
        """
        state(t), action(t), reward(t), state(t+1)の遷移を行う
        action が確率的行動の場合を考慮し、引数のaction が入力された場合は
        その行動を確定的に行い、Noneの場合は action() に従う事とする
        """
        self.transition_before()
        self.step += 1 # step を追加する
        self.action_prev = action if action is not None else self.action()
        self.state_prev  = self.state_now
        self.transition_middle()
        self.state_now   = self.state()
        self.reward_prev = self.reward()
        self.transition_after()
    
    def is_finish(self) -> bool:
        raise NotImplementedError()
 
    def q_update(self):
        """
        Q値の更新を行う
        Q(s, a) = Q(s, a) + alpha*(r+gamma*maxQ(s')-Q(s, a))
        """
        logger.debug(f"state_prev: {self.state_prev}, action_prev: {self.action_prev}, reward_prev: {self.reward_prev}, state_now: {self.state_now}", color=["YELLOW"])
        self.qtable.update(self.state_prev, self.action_prev, self.reward_prev, self.state_now, prob_actions=self.prob_actions, on_episode=(self.is_finish() == False))
        
    def train(self, n_episode: int=100):
        """
        学習する
        """
        for i in range(n_episode):
            logger.info(f"episode: {self.episode}", color=["BOLD", "BLUE"])
            self.init()
            while self.is_finish() == False:
                self.transition()
                self.q_update()
            self.episode += 1


class StateManager(object):
    """
    状態管理のためのclass
    DQNにおいては、状態の数(input node数)はQ-tableのように全ての組み合わせの数にならない。
    例えば、行った国の状態をbit で表すと、(3カ国の場合)000,001,010,100,011,101,110,111のように2^3となるが、node数では3 nodeで表せる
    """
    def __init__(self):
        super(StateManager, self).__init__()
        self.dict_state = {}
        self.list_names = []
    
    def __len__(self):
        count = 0
        for x in self.list_names:
            if self.dict_state[x]["type"] == "onehot":
                count += self.dict_state[x]["nclass"]
            else:
                count += 1
        return count
    
    def set_state(self, name: str, state_type: str, state_list: List[object] = None):
        """
        Params::
            name: state の名称
            state_type: numeric, list, binary, onehot
                ※ binary の場合、その値が入ってこなければ 0 を入力する
            state_list: str, onehot の場合は必要
        """
        self.list_names.append(name)
        state = {}
        state["type"]   = state_type
        state["state"]  = {x:i for i, x in enumerate(state_list)} if state_list is not None else {}
        state["index"]  = {i:x for i, x in enumerate(state_list)} if state_list is not None else {}
        state["nclass"] = len(state_list) if state_list is not None else 0
        self.dict_state[name] = state
    
    def pattern(self) -> List[Tuple[object]]:
        """
        Return::
            考えられる組み合わせの状態を全て返却する. numeric や onehot の場合は error とする
        """
        listwk = []
        for x in self.list_names:
            if   self.dict_state[x]["type"] == "list":
                listwk.append(list(self.dict_state[x]["state"].keys()))
            elif self.dict_state[x]["type"] == "binary":
                listwk.append([0, 1])
            elif self.dict_state[x]["type"] == "onehot":
                raise Exception(f'We can not calculate {self.dict_state[x]["type"]} type.')
            elif self.dict_state[x]["type"] == "numeric":
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
    
    def conv(self, values) -> List[object]:
        output = np.zeros(0).astype(float)
        is_dict = (type(values) == dict)
        for i, x in enumerate(self.list_names):
            val = values.get(x) if is_dict else values[i]
            if   self.dict_state[x]["type"] == "numeric":
                output = np.append(output, val)
            elif self.dict_state[x]["type"] == "list":
                output = np.append(output, self.dict_state[x]["state"][val])
            elif self.dict_state[x]["type"] == "onehot":
                val = np.identity(self.dict_state[x]["nclass"])[self.dict_state[x]["state"][val]]
                output = np.append(output, val)
            elif self.dict_state[x]["type"] == "binary":
                val = False if val is None else val
                output = np.append(output, bool(val))
        return output.astype(int)

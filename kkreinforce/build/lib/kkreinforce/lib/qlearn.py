import pandas as pd
import numpy as np
from typing import List

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
        self.dict_state  = {x:i for i, x in enumerate(self.q.index)}
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


    def get_max(self, state: object, list_action_flg: List[bool] = None):
        se_qvalue = self.q.iloc[self.dict_state[state], :]
        if list_action_flg is not None: se_qvalue = se_qvalue.loc[list_action_flg]
        action    = se_qvalue.idxmax()
        if type(action) == np.float and np.isnan(action): action = self.q.columns[list_action_flg][0]
        val = se_qvalue.max()
        val = 0 if np.isnan(val) else val
        return val, action


    def update(self, state, action, reward, state_next):
        """
        Q table の更新を行う
        """
        # Q(s, a) = Q(s, a) + alpha*(r+gamma*maxQ(s')-Q(s, a))
        q     = self.get_value(state, action=action) # Q(s, a)
        max_q = self.get_max(state_next)[0]
        q_update = (self.alpha * (reward + (self.gamma * max_q) - q))
        logger.debug(f'q now: {q}, max q: {max_q}, reward: {reward}, update q: {q_update}')

        # 参照形式で更新する
        self.ndf[self.dict_state[state], self.dict_action[action]] = q + q_update

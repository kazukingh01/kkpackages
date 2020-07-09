import numpy as np
import folium
from typing import List

# local package
from kkreinforce.lib.kkrl import StateManager
from kkreinforce.lib.qlearn import QLearn, QTable
from kkreinforce.models.tsp.tsp_base import TSPModelBase, DivIcon
from kkimagemods.util.logger import set_logger, set_loglevel
logger = set_logger(__name__)


class TSPModel(TSPModelBase, QLearn):
    """
    巡回セールスマン問題を、Q学習で実装する
    状態:
        今いる国
    行動:
        次に行く国
    報酬:
        国と国の移動距離を損失として与える
    """
    def __init__(self, epsilon: float, alpha: float, gamma: float, file_csv: str="../data/s59h30megacities_utf8.csv", n_capital: int=None):
        # まずは Base class で初期化して, df を load
        TSPModelBase.__init__(self, file_csv=file_csv, n_capital=n_capital)

        # Action の定義
        self.list_action = np.random.permutation(self.df["capital_en"].unique())

        # QTable の定義
        qfunc = QTable(state_list=self.list_action, action_list=self.list_action, alpha=alpha, gamma=gamma)
        QLearn.__init__(self, qfunc=qfunc, epsilon=epsilon)
        self.action_pprev  = None

        # 巡回できるようにするためのパラメータ
        self.is_back       = False
        self.first_country = None


    def initialize(self):
        """
        episode単位の初期化
        """
        self.action_prev  = "Tokyo"
        self.state_now    = self.action_prev
        self.state_prev   = self.action_prev
        self.reward_now   = 0
        self.prob_actions = np.ones_like(self.list_action).astype(bool)

        self.action_pprev  = self.action_prev
        self.country_prev  = self.action_prev
        self.country_now   = self.action_prev
        self.loss          = 0
        self.is_back       = False
        self.first_country = self.action_prev
        self.df.to_csv("now_setting.csv")


    def state(self, state_prev: object=None, action_prev: object=None) -> object:
        """
        state_prevからpolicyに従ったactionを行い、次のstateを決定する
        """
        action_prev = self.action_prev if action_prev is None else action_prev
        return action_prev


    def action_best(self) -> object:
        """
        state_now から決定される最適 action を返却する
        """
        _, action = self.qfunc.get_max(self.state_now, prob_actions=self.prob_actions)
        return action


    def action_random(self) -> object:
        """
        state_now から決定される random action を返却する
        """
        ndf = self.list_action[self.prob_actions]
        return np.random.permutation(ndf)[0]


    def reward(self, state_prev: object=None, action_prev: object=None, state_now: object=None) -> object:
        """
        state_prev と action_prev (と state_now) から決まる reward を返却する
        """
        action_prev = self.action_prev if action_prev is None else action_prev 
        dist = self.distance(self.action_pprev, action_prev)
        if self.is_eval: self.loss += dist
        return -1 * dist


    def transition_before_all(self):
        self.action_pprev  = self.action_prev


    def transition_after_state(self):
        self.prob_actions[self.list_action == self.action_prev] = False


    def transition_after_all(self):
        self.country_prev = self.country_now
        self.country_now  = self.action_prev


    def is_finish(self) -> bool:
        if self.is_back and (self.prob_actions == True).sum() == 0:
            return True
        if (self.prob_actions == True).sum() == 0:
            self.is_back = True
            self.prob_actions[self.list_action == self.first_country] = True # 最初の国を行けるようにする
        return False


    def q_update(self):
        """
        Q値の更新を行う
        Q(s, a) = Q(s, a) + alpha*(r+gamma*maxQ(s')-Q(s, a))
        """
        logger.debug(f"state_prev: {self.state_prev}, action_prev: {self.action_prev}, reward_now: {self.reward_now}, state_now: {self.state_now}", color=["YELLOW"])
        self.qfunc.update(self.state_prev, self.action_prev, self.reward_now, self.state_now, prob_actions=self.prob_actions, on_episode=(self.is_finish() == 0))


    """ ※ここから独自関数※ """

    def play(self, output: str="result.html", set_actions: List[object] = None):
        """
        学習した結果のplay
        """
        world_map = folium.Map() # 世界地図の作成
        self.init()
        self.is_eval = True
        lat_s, lon_s = self.get_lat_lon(self.country_now)
        folium.Marker(location=[lat_s, lon_s], popup=self.country_now).add_to(world_map)
        i = 0
        while self.is_finish() == 0:
            if set_actions is not None:
                action = set_actions[i]
                i += 1
            else:
                action = self.action_best()
            dist   = self.distance(self.country_now, action)
            lat_s, lon_s = self.get_lat_lon(self.country_now)
            lat_e, lon_e = self.get_lat_lon(action)
            folium.Marker(location=[lat_e, lon_e], popup=folium.Popup(html=str(self.step), max_width="50%", show=True), tooltip=action).add_to(world_map)
            folium.PolyLine(locations=[[lat_s, lon_s], [lat_e, lon_e]], popup=folium.Popup(html=str(round(dist, 3)), max_width="50%", show=True), weight=1).add_to(world_map)
            self.transition(action=action)
            logger.info(f'country: {self.country_prev}, action: {self.action_prev}', color=["BOLD", "BLUE"])

        folium.map.Marker(self.get_lat_lon(action),
            icon=DivIcon(size=(150,36), anchor=(150,0), html=str(self.loss),
            style="""
                font-size:36px;
                background-color: transparent;
                border-color: transparent;
                text-align: right;
                """
            )
        ).add_to(world_map)
        world_map.save(output)
        logger.info(f'finish !! all distance: {self.loss}', color=["BOLD", "GREEN"])



class TSPModel2(TSPModel):
    """
    巡回セールスマン問題を、Q学習で実装する.
    ※行動の履歴を考慮する
    状態:
        今いる国 + 過去に行った国の履歴
    行動:
        次に行く国
    報酬:
        国と国の移動距離を損失として与える
    """
    def __init__(self, epsilon: float, alpha: float, gamma: float, file_csv: str="../data/s59h30megacities_utf8.csv", n_capital: int=10):
        # まずは Base class で初期化して, df を load
        TSPModelBase.__init__(self, file_csv=file_csv, n_capital=n_capital)

        # Action の定義
        self.list_action = np.random.permutation(self.df["capital_en"].unique())

        # State の定義
        self.state_mng = StateManager()
        self.state_mng.set_state("country", state_type="list", state_list=self.list_action)
        for x in self.list_action:
            ## 国の滞在履歴を状態に組み込む
            self.state_mng.set_state(x, state_type="binary", state_list=None)

        # QTable の定義
        qfunc = QTable(state_list=self.state_mng.pattern(), action_list=self.list_action, alpha=alpha, gamma=gamma)
        QLearn.__init__(self, qfunc=qfunc, epsilon=epsilon)


    def initialize(self):
        """
        episode単位の初期化
        """
        self.state_mng.reset_value()
        init_action = np.random.permutation(self.list_action)[0]
        self.state_mng.set_values({"country": init_action, init_action:True})
        self.state_now    = tuple(self.state_mng.conv().tolist())
        self.state_prev   = self.state_now
        self.action_prev  = init_action
        self.reward_now   = 0
        self.prob_actions = np.ones_like(self.list_action).astype(bool)

        self.action_pprev = self.action_prev
        self.country_prev = init_action
        self.country_now  = init_action
        self.loss         = 0
        self.is_back       = False
        self.first_country = self.action_prev
        self.df.to_csv("now_setting.csv")


    def state(self, state_prev: object=None, action_prev: object=None) -> object:
        """
        state_prevからpolicyに従ったactionを行い、次のstateを決定する
        """
        action_prev = self.action_prev if action_prev is None else action_prev
        return tuple(self.state_mng.conv_tmp({"country": action_prev, action_prev:True}).tolist())


    def transition_after_state(self):
        super().transition_after_state()
        self.state_mng.set_values({"country": self.action_prev, self.action_prev:True})
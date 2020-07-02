import torch
import numpy as np
import folium
from typing import List

# local package
from kkreinforce.lib.kkrl import StateManager
from kkreinforce.lib.kknn import TorchNN, Layer
from kkreinforce.lib.policygrad import PolicyGradient, PolicyGradientNN
from kkreinforce.models.tsp.tsp_base import TSPModelBase, DivIcon
from kkimagemods.util.logger import set_logger, set_loglevel
logger = set_logger(__name__)



class TSPModel(TSPModelBase, PolicyGradient):
    """
    巡回セールスマン問題を、方策勾配法で実装する
    状態:
        今いる国＋過去に行った国の履歴
    行動:
        次に行く国
    報酬:
        国を一周回ったときに報酬
    """
    def __init__(self, epsilon: float, alpha: float, gamma: float, file_csv: str="../data/s59h30megacities_utf8.csv", n_capital: int=None):
        # まずは Base class で初期化して, df を load
        TSPModelBase.__init__(self, file_csv=file_csv, n_capital=n_capital)

        # Action の定義
        self.list_action = np.random.permutation(self.df["capital_en"].unique())

        # State の定義
        self.state_mng = StateManager()
        self.state_mng.set_state("country", state_type="onehot",        state_list=self.list_action)
        self.state_mng.set_state("history", state_type="onehot_binary", state_list=self.list_action) # 国の滞在履歴を状態に組み込む

        # NN の定義
        torch_nn = TorchNN(len(self.state_mng),
            Layer("fc1",   torch.nn.Linear,      128,  None, (), {}),
            Layer("norm1", torch.nn.BatchNorm1d, 0,    None, (), {}),
            Layer("relu1", torch.nn.ReLU,        None, None, (), {}),
            Layer("fc2",   torch.nn.Linear,      len(self.list_action), None, (), {}),
            Layer("soft",  torch.nn.Softmax,     None, None, (), {"dim":1}),
        )
        policy_nn = PolicyGradientNN(torch_nn, self.list_action, 128, 1000, unit_memory=None, lr=0.01)
        PolicyGradient.__init__(self, policy=policy_nn, list_action=self.list_action)
        self.action_pprev  = None

        # 巡回できるようにするためのパラメータ
        self.is_back       = False
        self.first_country = None


    def initialize(self):
        """
        episode単位の初期化
        """
        init_action       = self.list_action[0]
        self.state_mng.reset_value()
        self.state_mng.set_value("country", init_action)
        self.state_mng.set_value("history", init_action)
        self.action_prev  = init_action
        self.state_now    = self.state_mng.conv().copy()
        self.state_prev   = self.state_now
        self.reward_now   = 0

        self.prob_actions  = np.ones_like(self.list_action).astype(bool)
        self.prob_actions[self.list_action == self.action_prev] = False
        self.country_prev  = self.action_prev
        self.country_now   = self.action_prev
        self.action_pprev  = self.action_prev
        self.loss          = 0
        self.is_back       = False
        self.first_country = self.action_prev
        self.df.to_csv("now_setting.csv")


    def state(self, state_prev: object=None, action_prev: object=None) -> object:
        """
        state_prev と action_prev から決まる state を返却する
        """
        action_prev = self.action_prev if action_prev is None else action_prev
        state_prev  = self.state_prev  if state_prev  is None else state_prev
        return self.state_mng.conv_tmp({"country": action_prev, "history":action_prev})


    def reward(self, state_prev: object=None, action_prev: object=None, state_now: object=None) -> object:
        """
        state_prev と action_prev (と state_now) から決まる reward を返却する
        ※報酬は定義的にはs(t)とa(t)に対して決まると思っているが、s(t+1)から受け取る解釈もあるので、念の為付け加える
        """
        action_prev = self.action_prev if action_prev is None else action_prev
        dist = self.distance(self.action_pprev, action_prev)
        self.loss += dist
        return 10 if self.is_finish() else 0
        #return -1 * self.loss if self.is_finish() else 0


    def transition_before_all(self):
        self.action_pprev = self.action_prev


    def transition_after_state(self):
        self.prob_actions[self.list_action == self.action_prev] = False
        self.state_mng.set_value("country", self.action_prev)
        self.state_mng.set_value("history", self.action_prev)


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
        while self.is_finish() == False:
            if set_actions is not None:
                action = set_actions[i]
                i += 1
            else:
                action = self.action()
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

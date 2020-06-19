from typing import List
import pandas as pd
import numpy as np
import folium
import torch
pd.options.display.max_rows = 100

# local package
from kkreinforce.lib.qlearn import QTable, QLearn, StateManager
from kkreinforce.lib.dqn import DQN, TorchNN, Layer
from kkimagemods.util.logger import set_logger, set_loglevel
logger = set_logger(__name__)


def sigmoid(a):
    return 1 / (1 + np.exp(-a))

def cal_rho(lon_a,lat_a,lon_b,lat_b):
    ra=6378.140  # equatorial radius (km)
    rb=6356.755  # polar radius (km)
    F=(ra-rb)/ra # flattening of the earth
    rad_lat_a=np.radians(lat_a)
    rad_lon_a=np.radians(lon_a)
    rad_lat_b=np.radians(lat_b)
    rad_lon_b=np.radians(lon_b)
    pa=np.arctan(rb/ra*np.tan(rad_lat_a))
    pb=np.arctan(rb/ra*np.tan(rad_lat_b))
    xx=np.arccos(np.sin(pa)*np.sin(pb)+np.cos(pa)*np.cos(pb)*np.cos(rad_lon_a-rad_lon_b))
    c1=(np.sin(xx)-xx)*(np.sin(pa)+np.sin(pb))**2/np.cos(xx/2)**2
    c2=(np.sin(xx)+xx)*(np.sin(pa)-np.sin(pb))**2/np.sin(xx/2)**2
    dr=F/8*(c1-c2)
    rho=ra*(xx+dr)
    return float(rho)


from folium import MacroElement
from folium.features import Template
class DivIcon(MacroElement):
    def __init__(self, html='', size=(30,30), anchor=(0,0), style=''):
        """TODO : docstring here"""
        super(DivIcon, self).__init__()
        self._name = 'DivIcon'
        self.size = size
        self.anchor = anchor
        self.html = html
        self.style = style

        self._template = Template(u"""
            {% macro header(this, kwargs) %}
              <style>
                .{{this.get_name()}} {
                    {{this.style}}
                    }
              </style>
            {% endmacro %}
            {% macro script(this, kwargs) %}
                var {{this.get_name()}} = L.divIcon({
                    className: '{{this.get_name()}}',
                    iconSize: [{{ this.size[0] }},{{ this.size[1] }}],
                    iconAnchor: [{{ this.anchor[0] }},{{ this.anchor[1] }}],
                    html : "{{this.html}}",
                    });
                {{this._parent.get_name()}}.setIcon({{this.get_name()}});
            {% endmacro %}
            """)


class TSPModelBase(object):
    def __init__(self, file_csv: str="../data/s59h30megacities_utf8.csv", n_capital: int=None):
        df = pd.read_csv(file_csv, sep="\t")
        df = df[df["iscapital"] == 1]
        df["capital_en"] = df["capital_en"].replace(r"\s", "_", regex=True)
        if n_capital is not None:
            ndf = np.random.permutation(df["capital_en"].unique())[:n_capital] # 都市を限定する
            df  = df[df["capital_en"].isin(ndf)]
        self.df = df.copy()

    def distance(self, city1: str, city2: str) -> float:
        """
        city1 から city2 への距離を計算する
        """
        if city1 == city2:
            val = 0
        else:
            df = self.df
            se1 = df[df["capital_en"] == city1].iloc[0]
            se2 = df[df["capital_en"] == city2].iloc[0]
            val = cal_rho(se1["lon"], se1["lat"], se2["lon"], se2["lat"]) / 1000.
        return val
    
    def get_lat_lon(self, city: str) -> (float, float, ):
        df = self.df
        return df[df["capital_en"] == city].iloc[0][["lat", "lon"]].values



class TSPModel(TSPModelBase, QLearn):
    def __init__(self, epsilon: float, alpha: float, gamma: float, file_csv: str="../data/s59h30megacities_utf8.csv", n_capital: int=None):
        # まずは Base class で初期化して, df を load
        TSPModelBase.__init__(self, file_csv=file_csv, n_capital=n_capital)

        # QTable の定義
        self.list_action = np.random.permutation(self.df["capital_en"].unique())
        qtable = QTable(state_list=self.list_action, action_list=self.list_action, alpha=alpha, gamma=gamma)
        QLearn.__init__(self, qtable=qtable, epsilon=epsilon)


    def initialize(self):
        """
        episode単位の初期化
        """
        self.action_prev  = "Tokyo"
        self.state_now    = self.action_prev
        self.state_prev   = self.action_prev
        self.reward_prev  = 0
        self.prob_actions = np.ones_like(self.list_action).astype(bool)

        self.country_prev = self.action_prev
        self.country_now  = self.action_prev
        self.loss         = 0
        self.transition(action=self.action_prev)


    def state(self, state_prev: object=None, action_prev: object=None) -> object:
        """
        state_prev と action_prev から決まる state を返却する
        """
        action_prev = self.action_prev if action_prev is None else action_prev
        return action_prev


    def action_best(self) -> object:
        """
        state_now から決定される最適 action を返却する
        """
        _, action = self.qtable.get_max(self.state_now, prob_actions=self.prob_actions)
        return action


    def action_random(self) -> object:
        """
        state_now から決定される random action を返却する
        """
        ndf = self.list_action[self.prob_actions]
        return np.random.permutation(ndf)[0]


    def reward(self, state_prev: object=None, action_prev: object=None) -> object:
        """
        state_prev と action_prev から決まる reward を返却する
        """
        state_prev  = self.state_prev  if state_prev  is None else state_prev 
        action_prev = self.action_prev if action_prev is None else action_prev 
        dist = self.distance(state_prev, action_prev)
        if self.is_eval: self.loss += dist
        return -1 * dist


    def transition_middle(self):
        self.prob_actions[self.list_action == self.action_prev] = False


    def transition_after(self):
        self.country_prev = self.country_now
        self.country_now  = self.action_prev


    def is_finish(self) -> bool:
        if (self.prob_actions == True).sum() == 0:
            return True
        else:
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
        qtable = QTable(state_list=self.state_mng.pattern(), action_list=self.list_action, alpha=alpha, gamma=gamma)
        QLearn.__init__(self, qtable=qtable, epsilon=epsilon)


    def initialize(self):
        """
        episode単位の初期化
        """
        init_action = np.random.permutation(self.list_action)[0]
        init_state  = tuple([init_action] +  [0 for _ in self.list_action])
        self.state_now    = init_state
        self.state_prev   = init_state
        self.action_prev  = init_action
        self.reward_prev  = 0
        self.prob_actions = np.ones_like(self.list_action).astype(bool)

        self.country_prev = init_action
        self.country_now  = init_action
        self.loss         = 0
        self.transition(action=self.action_prev)


    def state(self, state_prev: object=None, action_prev: object=None) -> object:
        """
        state_prev と action_prev から決まる state を返却する
        """
        state_prev  = self.state_prev  if state_prev  is None else state_prev 
        action_prev = self.action_prev if action_prev is None else action_prev
        statewk    = list(state_prev)
        statewk[0] = action_prev
        statewk[np.where(self.list_action == action_prev)[0].min() + 1] = 1
        return tuple(statewk)


    def reward(self, state_prev: object=None, action_prev: object=None) -> object:
        """
        state_prev と action_prev から決まる reward を返却する
        """
        state_prev  = self.state_prev  if state_prev  is None else state_prev 
        action_prev = self.action_prev if action_prev is None else action_prev 
        dist = self.distance(state_prev[0], action_prev)
        if self.is_eval: self.loss += dist
        return -1 * dist



class TSPModel3(TSPModel):
    def __init__(self, epsilon: float, alpha: float, gamma: float, file_csv: str="../data/s59h30megacities_utf8.csv", n_capital: int=None):
        # まずは Base class で初期化して, df を load
        TSPModelBase.__init__(self, file_csv=file_csv, n_capital=n_capital)

        # Action の定義
        self.list_action = np.random.permutation(self.df["capital_en"].unique())

        # State の定義
        self.state_mng = StateManager()
        self.state_mng.set_state("country", state_type="onehot", state_list=self.list_action)

        # DQN の定義
        torch_nn = TorchNN(len(self.list_action), 
            Layer("fc1",   torch.nn.Linear,      128,  None, (), {}),
            Layer("norm1", torch.nn.BatchNorm1d, 0,    None, (), {}),
            Layer("relu1", torch.nn.ReLU,        None, None, (), {}),
            Layer("fc2",   torch.nn.Linear,      128,  None, (), {}),
            Layer("norm2", torch.nn.BatchNorm1d, 0,    None, (), {}),
            Layer("relu2", torch.nn.ReLU,        None, None, (), {}),
            Layer("fc3",   torch.nn.Linear,      len(self.list_action), None, (), {}),
        )
        qtable = DQN(torch_nn, self.list_action, alpha=alpha, gamma=gamma, batch_size=128, capacity=1000)
        QLearn.__init__(self, qtable=qtable, epsilon=epsilon)

        # 巡回できるようにするためのパラメータ
        self.is_back = False
        self.first_country = self.action_prev


    def initialize(self):
        """
        episode単位の初期化
        """
        self.action_prev  = "Tokyo"
        self.state_now    = self.state_mng.conv([self.action_prev])
        self.state_prev   = self.state_now
        self.reward_prev  = 0
        self.prob_actions = np.ones_like(self.list_action).astype(bool)

        self.country_prev = self.action_prev
        self.country_now  = self.action_prev
        self.loss         = 0
        self.transition(action=self.action_prev)

        self.is_back = False
        self.first_country = self.action_prev
        self.df.to_csv("now_setting.csv")


    def is_finish(self) -> bool:
        if self.is_back and (self.prob_actions == True).sum() == 0:
            return True
        if (self.prob_actions == True).sum() == 0:
            self.is_back = True
            self.prob_actions[self.list_action == self.first_country] = True # 最初の国を行けるようにする
        return False


    def state(self, state_prev: object=None, action_prev: object=None) -> object:
        """
        state_prev と action_prev から決まる state を返却する
        """
        action_prev = self.action_prev if action_prev is None else action_prev
        return self.state_mng.conv([action_prev])


    def reward(self, state_prev: object=None, action_prev: object=None) -> object:
        """
        state_prev と action_prev から決まる reward を返却する
        """
        state_prev  = self.state_prev  if state_prev  is None else state_prev
        action_prev = self.action_prev if action_prev is None else action_prev
        dist = self.distance(self.list_action[state_prev[:len(self.list_action)].astype(bool)][0], action_prev)
        if self.is_eval: self.loss += dist
        return -1 * dist



class TSPModel4(TSPModel3):
    def __init__(self, epsilon: float, alpha: float, gamma: float, file_csv: str="../data/s59h30megacities_utf8.csv", n_capital: int=None):
        # まずは Base class で初期化して, df を load
        TSPModelBase.__init__(self, file_csv=file_csv, n_capital=n_capital)

        # Action の定義
        self.list_action = np.random.permutation(self.df["capital_en"].unique())

        # State の定義
        self.state_mng = StateManager()
        self.state_mng.set_state("country", state_type="onehot", state_list=self.list_action)
        for x in self.list_action:
            ## 国の滞在履歴を状態に組み込む
            self.state_mng.set_state(x, state_type="numeric", state_list=None)

        # DQN の定義
        torch_nn = TorchNN(len(self.state_mng), 
            Layer("fc1",   torch.nn.Linear,      128,  None, (), {}),
            Layer("norm1", torch.nn.BatchNorm1d, 0,    None, (), {}),
            Layer("relu1", torch.nn.ReLU,        None, None, (), {}),
            Layer("fc2",   torch.nn.Linear,      256,  None, (), {}),
            Layer("norm2", torch.nn.BatchNorm1d, 0,    None, (), {}),
            Layer("relu2", torch.nn.ReLU,        None, None, (), {}),
            Layer("fc3",   torch.nn.Linear,      128,  None, (), {}),
            Layer("norm3", torch.nn.BatchNorm1d, 0,    None, (), {}),
            Layer("relu3", torch.nn.ReLU,        None, None, (), {}),
            Layer("fc4",   torch.nn.Linear,      len(self.list_action), None, (), {}),
        )
        qtable = DQN(torch_nn, self.list_action, alpha=alpha, gamma=gamma, batch_size=128, capacity=1000)
        QLearn.__init__(self, qtable=qtable, epsilon=epsilon)

        # 状態の追加
        self.action_hist_sum = None

        # 巡回できるようにするためのパラメータ
        self.is_back = False
        self.first_country = self.action_prev


    def initialize(self):
        """
        episode単位の初期化
        """
        init_action       = self.list_action[0]
        self.action_prev  = init_action
        self.action_hist_sum = np.zeros_like(self.list_action).astype(float)
        self.state_now    = self.state_mng.conv([self.action_prev] + [True for _ in self.list_action])
        self.state_prev   = self.state_now
        self.reward_prev  = 0
        self.prob_actions = np.ones_like(self.list_action).astype(bool)

        self.country_prev = self.action_prev
        self.country_now  = self.action_prev
        self.loss         = 0
        self.transition(action=self.action_prev)

        self.is_back = False
        self.first_country = self.action_prev
        self.df.to_csv("now_setting.csv")        


    def state(self, state_prev: object=None, action_prev: object=None) -> object:
        """
        state_prev と action_prev から決まる state を返却する
        """
        action_prev = self.action_prev if action_prev is None else action_prev
        _state = self.state_mng.conv([action_prev] + self.action_hist_sum.tolist())
        return _state


    def reward(self, state_prev: object=None, action_prev: object=None) -> object:
        """
        state_prev と action_prev から決まる reward を返却する
        """
        state_prev  = self.state_prev  if state_prev  is None else state_prev
        action_prev = self.action_prev if action_prev is None else action_prev
        dist = self.distance(self.list_action[state_prev[:len(self.list_action)].astype(bool)][0], action_prev)
        self.loss += dist
        return -1 * self.loss / 10 if self.is_finish() else 0
    

    def transition_middle(self):
        super().transition_middle()
        self.action_hist_sum = self.action_hist_sum + (self.prob_actions == False).astype(float) # 行った国の履歴はどんどん足し上げる. 数字が大きいほど過去に行ったことを表現



class TSPModel5(TSPModel4):
    def __init__(self, epsilon: float, alpha: float, gamma: float, file_csv: str="../data/s59h30megacities_utf8.csv", n_capital: int=None):
        # まずは Base class で初期化して, df を load
        TSPModelBase.__init__(self, file_csv=file_csv, n_capital=n_capital)

        # Action の定義
        self.list_action = np.random.permutation(self.df["capital_en"].unique())

        # State の定義
        self.state_mng = StateManager()
        self.state_mng.set_state("country", state_type="onehot", state_list=self.list_action)
        for x in self.list_action:
            ## 国の滞在履歴を状態に組み込む
            self.state_mng.set_state(x, state_type="numeric", state_list=None)

        # DQN の定義
        torch_nn = TorchNN(len(self.state_mng), 
            Layer("lstm",  torch.nn.LSTM,        128,  "rnn_all", (), {}),
            Layer("norm1", torch.nn.BatchNorm1d, 0,    None,      (), {}),
            Layer("relu1", torch.nn.ReLU,        None, None,      (), {}),
            Layer("fc2",   torch.nn.Linear,      128,  None,      (), {}),
            Layer("norm2", torch.nn.BatchNorm1d, 0,    None,      (), {}),
            Layer("relu2", torch.nn.ReLU,        None, None,      (), {}),
            Layer("fc3",   torch.nn.Linear,      64,   None,      (), {}),
            Layer("norm3", torch.nn.BatchNorm1d, 0,    None,      (), {}),
            Layer("relu3", torch.nn.ReLU,        None, None,      (), {}),
            Layer("fc4",   torch.nn.Linear,      len(self.list_action), None, (), {}),
        )
        qtable = DQN(torch_nn, self.list_action, alpha=alpha, gamma=gamma, batch_size=5, capacity=5, unit_memory="episode", lr=0.001)
        QLearn.__init__(self, qtable=qtable, epsilon=epsilon)

        # 状態の追加
        self.action_hist_sum = None
        self.state_history = []

        # 巡回できるようにするためのパラメータ
        self.is_back = False
        self.first_country = self.action_prev


    def initialize(self):
        self.state_history = []
        super().initialize()


    def state(self, state_prev: object=None, action_prev: object=None) -> object:
        _state = super().state()
        self.state_history.append(_state)
        return _state


    def action_best(self) -> object:
        """
        state_now から決定される最適 action を返却する
        ※ここでは state_now を含む state_history を使用する
        """
        _, action = self.qtable.get_max(np.array(self.state_history), prob_actions=self.prob_actions)
        return action



class TSPModel6(TSPModel5):
    def __init__(self, epsilon: float, alpha: float, gamma: float, file_csv: str="../data/s59h30megacities_utf8.csv", n_capital: int=None):
        super().__init__(epsilon=epsilon, alpha=alpha, gamma=gamma, file_csv=file_csv, n_capital=n_capital)
        self.loss_max = None

    def reward(self, state_prev: object=None, action_prev: object=None) -> object:
        #state_prev と action_prev から決まる reward を返却する
        state_prev  = self.state_prev  if state_prev  is None else state_prev
        action_prev = self.action_prev if action_prev is None else action_prev
        dist = self.distance(self.list_action[state_prev[:len(self.list_action)].astype(bool)][0], action_prev)
        self.loss += dist
        dist = 1./dist if dist > 0 else 0
        r = None
        if self.is_finish() and self.is_eval == False:
            if self.loss_max is None:
                self.loss_max = self.loss
            else:
                if self.loss_max >= self.loss:
                    # 距離が短かったら更新する
                    self.loss_max = self.loss
            r = (self.loss_max / self.loss)
            if   r > 0.99: r = 10
            elif r > 0.9:  r = 5
            elif r > 0.8:  r = 2
            else:          r = 0
        return r if self.is_finish() else 0

    def play(self, output: str="result.html"):
        best_actions = None
        super().play(output=output, set_actions=None)
        if len(self.qtable.memory.memory) > 0:
            _, best_actions, _, _, _ = self.qtable.memory.sample(indexes=[0])
            best_actions = best_actions.reshape(-1)
        super().play(output=output+".best.html", set_actions=best_actions)
        logger.info(f"loss_max: {self.loss_max}")

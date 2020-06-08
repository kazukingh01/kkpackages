from typing import List
import pandas as pd
import numpy as np
import folium

# local package
from kkimagemods.util.logger import set_logger, set_loglevel
logger = set_logger(__name__)
def log_green(  msg: str): logger.info(msg, color=["BOLD", "GREEN"])
def log_blue(   msg: str): logger.info(msg, color=["BOLD", "BLUE"])
def log_yellow( msg: str): logger.info(msg, color=["BOLD", "YELLOW"])


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


class TSPModel:
    """
    巡回セールスマン問題に対する強化学習モデルのクラス
    全ての都市を訪問するまでの総合距離を最小化するアプローチを取る
    """

    def __init__(self, epsilon, alpha, gamma):
        df = pd.read_csv("../data/s59h30megacities_utf8.csv", sep="\t")
        df = df[df["iscapital"] == 1]
        df["capital_en"] = df["capital_en"].replace(r"\s", "_", regex=True)

        self.df = df.copy()

        # action は ある都市に行く行為なので、全ての都市の名前を入れる
        self.list_action = np.random.permutation(df["capital_en"].unique())
        # Q table の作成
        self.df_qvalue = pd.DataFrame(0.0, index=self.list_action.copy(), columns=self.list_action.copy(), dtype=float)

        # init 内で初期化する値. 敢えてNone
        self.list_action_flg = None
        self.state           = None
        self.is_finish       = False
        self.loss            = None
        self.step            = None

        # ハイパーパラメータ
        self.epsilon = epsilon # greedy 行動の閾値
        self.alpha = alpha
        self.gamma = gamma

        # 初期化
        self.init()


    def init(self):
        """
        episode単位の初期化
        """
        log_green("Initialize")
        self.loss  = 0
        self.step  = 0
        self.state = "Tokyo"
        self.is_finish = False
        self.list_action_flg = np.ones_like(self.list_action).astype(bool)
        self.transition("Tokyo") # 初期値は東京
    

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


    def action(self):
        # 一度訪問した箇所はaction listから外す
        list_action = self.list_action[self.list_action_flg]
        action, se_qvalue = None, None
        # greedy選択
        if np.random.uniform() < self.epsilon:  # random行動
            # 全て同じ確率のもと一つ選択する
            action = list_action[np.random.permutation(np.arange(list_action.shape[0]))[0]]
        else:
            se_qvalue = self.df_qvalue.loc[self.state, self.list_action_flg]
            action = se_qvalue.idxmax()
        logger.debug(f'action: {action}. max Q value: {"random" if se_qvalue is None else se_qvalue.max()}')
        return action


    def reward(self, action: str):
        """
        あるstateでactionした時に得られる報酬と行動の結果を定義
        ここでは移動した距離の合計を損失という形の報酬で与える
        """
        distance = -1 * self.distance(self.state, action)
        return distance #self.loss + distance if self.loss is not None else distance


    def transition(self, action: str):
        """
        action の結果得られる情報の更新を行う
        """
        self.step += 1
        # 距離の合計を計算する
        self.loss  = self.reward(action)
        self.state = action # 現在の地点をaction地点に
        self.list_action_flg[np.argmax(self.list_action == action)] = False #一度訪問した箇所はFalseに
        if action not in self.df_qvalue.index.tolist():
            # もし Q table の state がない場合は作成する
            self.df_qvalue.loc[action] = 0.0

        # もし、全ての都市を訪問し終えたら完了フラグを立てる
        if (self.list_action_flg == True).sum() == 0:
            self.is_finish = True
    

    def update(self, action: str):
        """
        Q値の更新を行う
        """
        # Q(s, a) = Q(s, a) + alpha*(r+gamma*maxQ(s')-Q(s, a))
        q        =     self.df_qvalue.loc[self.state, action] # Q(s, a)
        max_q    = max(self.df_qvalue.loc[action, :].values)  # 遷移先状態でのmaxQ値
        reward   = self.reward(action)
        q_update = (self.alpha * (reward + (self.gamma * max_q) - q))
        log_yellow(f'q now: {q}, reward: {reward}, update q: {q_update}')
        self.df_qvalue[self.state][action] = q + q_update
        

    def train(self, n_episode: int=100):
        """
        学習する
        """
        for i in range(n_episode):
            log_blue(f"episode: {i}")
            self.init()
            while self.is_finish == False:
                action = self.action()
                self.update(action)
                self.transition(action)


    def play(self, output: str="result.html"):
        """
        学習した結果のplay
        """
        world_map = folium.Map() # 世界地図の作成
        epsilon = self.epsilon
        self.epsilon = 0.0 # play 時は randmo な行動をなくす
        self.init()
        lat_s, lon_s = self.get_lat_lon(self.state)
        folium.Marker(location=[lat_s, lon_s], popup=self.state).add_to(world_map)
        while self.is_finish == False:
            lat_s, lon_s = self.get_lat_lon(self.state)
            action = self.action()
            lat_e, lon_e = self.get_lat_lon(action)
            folium.Marker(location=[lat_e, lon_e], popup=folium.Popup(html=str(self.step), max_width="50%", show=True), tooltip=action).add_to(world_map)
            folium.PolyLine(locations=[[lat_s, lon_s], [lat_e, lon_e]], weight=1).add_to(world_map)
            log_blue(f'state: {self.state}, action: {action}')
            self.transition(action)
        self.epsilon = epsilon
        world_map.save(output)
        log_green(f'finish !! loss: {self.loss}')
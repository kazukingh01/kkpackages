import torch
import numpy as np

# local package
from kkreinforce.lib.kkrl import StateManager
from kkreinforce.lib.kknn import TorchNN, Layer
from kkreinforce.lib.qlearn import QLearn, DQN
from kkreinforce.models.tsp.tsp_base import TSPModelBase, DivIcon
from kkreinforce.models.tsp.tsp_qtable import TSPModel
from kkimagemods.util.logger import set_logger, set_loglevel
logger = set_logger(__name__)



class TSPModel3(TSPModel):
    """
    巡回セールスマン問題を、DQNで実装する
    状態:
        今いる国
    行動:
        次に行く国
    報酬:
        国と国の移動距離を損失として与える
    """
    def __init__(self, epsilon: float, gamma: float, file_csv: str="../data/s59h30megacities_utf8.csv", n_capital: int=None):
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
        qfunc = DQN(torch_nn, self.list_action, gamma=gamma, batch_size=128, capacity=1000, lr=0.001)
        qfunc.to_cuda()
        QLearn.__init__(self, qfunc=qfunc, epsilon=epsilon)


    def initialize(self, init_country="Tokyo"):
        """
        episode単位の初期化
        """
        self.state_mng.reset_value()
        init_action = self.list_action[0] if init_country is None else init_country
        self.state_mng.set_values({"country": init_action})
        self.state_now    = self.state_mng.conv()
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
        state_prev と action_prev から決まる state を返却する
        """
        action_prev = self.action_prev if action_prev is None else action_prev
        return self.state_mng.conv_tmp({"country":action_prev})


    def transition_after_state(self):
        super().transition_after_state()
        self.state_mng.set_values({"country": self.action_prev})


    def q_update(self):
        """
        Q値の更新を行う
        """
        logger.debug(f"state_prev: {self.state_prev}, action_prev: {self.action_prev}, reward_now: {self.reward_now}, state_now: {self.state_now}", color=["YELLOW"])
        self.qfunc.store(self.state_prev, self.action_prev, self.reward_now, self.state_now, prob_actions=self.prob_actions, on_episode=(self.is_finish() == False))
        self.qfunc.update()



class TSPModel4(TSPModel3):
    """
    巡回セールスマン問題を、DQNで実装する
    状態:
        今いる国 + 過去に行った国
    行動:
        次に行く国
    報酬:
        国と国の移動距離を損失として与える
    """
    def __init__(self, epsilon: float, gamma: float, file_csv: str="../data/s59h30megacities_utf8.csv", n_capital: int=None):
        # まずは Base class で初期化して, df を load
        TSPModelBase.__init__(self, file_csv=file_csv, n_capital=n_capital)

        # Action の定義
        self.list_action = np.random.permutation(self.df["capital_en"].unique())

        # State の定義
        self.state_mng = StateManager()
        self.state_mng.set_state("country", state_type="onehot",        state_list=self.list_action)
        self.state_mng.set_state("history", state_type="onehot_binary", state_list=self.list_action)            

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
        qfunc = DQN(torch_nn, self.list_action, gamma=gamma, batch_size=128, capacity=1000)
        qfunc.to_cuda()
        QLearn.__init__(self, qfunc=qfunc, epsilon=epsilon)


    def initialize(self, init_country="Tokyo"):
        """
        episode単位の初期化
        """
        self.state_mng.reset_value()
        init_action = self.list_action[0] if init_country is None else init_country
        self.state_mng.set_values({"country": init_action, "history": init_action})
        self.state_now    = self.state_mng.conv()
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
        state_prev と action_prev から決まる state を返却する
        """
        action_prev = self.action_prev if action_prev is None else action_prev
        return self.state_mng.conv_tmp({"country":action_prev, "history": action_prev})


    def transition_after_state(self):
        super().transition_after_state()
        self.state_mng.set_values({"country": self.action_prev, "history": self.action_prev})



class TSPModel5(TSPModel4):
    """
    巡回セールスマン問題を、DQN(LSTM)で実装する
    状態:
        今いる国 + 過去に行った国
    行動:
        次に行く国
    報酬:
        国と国の移動距離を損失として与える
    """
    def __init__(self, epsilon: float, gamma: float, file_csv: str="../data/s59h30megacities_utf8.csv", n_capital: int=None):
        super(TSPModel5, self).__init__(epsilon, gamma, file_csv=file_csv, n_capital=n_capital)

        # DQN の定義
        torch_nn = TorchNN(len(self.state_mng), 
            Layer("lstm",  torch.nn.LSTM,        128,  None,           (), {}),
            Layer("calc1", torch.nn.Identity,    None, "rnn_outonly",  (), {}),
            Layer("calc2", torch.nn.Identity,    None, "call_options", (), {}),
            Layer("calc3", torch.nn.Identity,    None, "rnn_all",      (), {}),
            Layer("norm1", torch.nn.BatchNorm1d, 0,    None,           (), {}),
            Layer("relu1", torch.nn.ReLU,        None, None,           (), {}),
            Layer("fc2",   torch.nn.Linear,      128,  None,           (), {}),
            Layer("norm2", torch.nn.BatchNorm1d, 0,    None,           (), {}),
            Layer("relu2", torch.nn.ReLU,        None, None,           (), {}),
            Layer("fc3",   torch.nn.Linear,      64,   None,           (), {}),
            Layer("norm3", torch.nn.BatchNorm1d, 0,    None,           (), {}),
            Layer("relu3", torch.nn.ReLU,        None, None,           (), {}),
            Layer("fc4",   torch.nn.Linear,      len(self.list_action), None, (), {}),
        )
        qfunc = DQN(torch_nn, self.list_action, gamma=gamma, batch_size=20, capacity=100, unit_memory="episode", lr=0.001)
        self.qfunc = qfunc
        self.qfunc.to_cuda()
        self.state_history = [] # LSTMなのでepisode中のstate履歴を記憶する


    def initialize(self, init_country=None):
        """
        episode単位の初期化
        """
        super().initialize(init_country=init_country)
        self.state_history = [self.state()]


    def action_best(self) -> object:
        """
        state_now から決定される最適 action を返却する
        ※ここでは state_now を含む state_history を使用する
        """
        _, action = self.qfunc.get_max(np.array(self.state_history), prob_actions=self.prob_actions)
        return action


    def transition_after_state(self):
        super().transition_after_state()
        self.state_history.append(super().state())


    def q_update(self):
        """
        Q値の更新を行う
        """
        logger.debug(f"state_prev: {self.state_prev}, action_prev: {self.action_prev}, reward_now: {self.reward_now}, state_now: {self.state_now}", color=["YELLOW"])
        self.qfunc.store(self.state_prev, self.action_prev, self.reward_now, self.state_now, prob_actions=self.prob_actions, on_episode=(self.is_finish() == False))
        if self.is_finish():
            self.qfunc.update()



class TSPModel6(TSPModel5):
    """
    巡回セールスマン問題を、DQN(LSTM)で実装する
    状態:
        今いる国 + 過去に行った国
    行動:
        次に行く国
    報酬:
        総合距離をマイナス損失として与える. それ以外は0
    """
    def __init__(self, epsilon: float, gamma: float, file_csv: str="../data/s59h30megacities_utf8.csv", n_capital: int=10):
        super().__init__(epsilon, gamma, file_csv=file_csv, n_capital=n_capital)
        self.qfunc.batch_size = 32
        self.qfunc.memory.capacity = 128
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
            else:          r = -2
        return r if self.is_finish() else 0

    def play(self, output: str="result.html"):
        best_actions = None
        super().play(output=output, set_actions=None)
        if len(self.qfunc.memory.memory) > 0:
            _, best_actions, _, _, _ = self.qfunc.memory.sample(indexes=[0])
            best_actions = best_actions.reshape(-1)
        super().play(output=output+".best.html", set_actions=best_actions)
        logger.info(f"loss_max: {self.loss_max}")



class TSPModel7(TSPModel):
    """
    巡回セールスマン問題を、DQNで実装する
    状態:
        今いる国の地図
    行動:
        次に行く国
    報酬:
        総合距離をマイナス損失として与える. それ以外は0
    """
    def __init__(self, epsilon: float, gamma: float, file_csv: str="../data/s59h30megacities_utf8.csv", n_capital: int=None):
        # まずは Base class で初期化して, df を load
        TSPModelBase.__init__(self, file_csv=file_csv, n_capital=n_capital)

        # Action の定義
        self.list_action = np.random.permutation(self.df["capital_en"].unique())

        # DQN の定義
        torch_nn = TorchNN(3, 
            Layer("conv1", torch.nn.Conv2d,      128,     None, (), {"kernel_size":5, "stride":5,}),
            Layer("relu1", torch.nn.ReLU,        None,    None, (), {}),
            Layer("pool1", torch.nn.MaxPool2d,   None,    None, (), {"kernel_size":2, "stride":2,}),
            Layer("conv2", torch.nn.Conv2d,      128,     None, (), {"kernel_size":3, "stride":3,}),
            Layer("relu2", torch.nn.ReLU,        None,    None, (), {}),
            Layer("pool2", torch.nn.MaxPool2d,   None,    None, (), {"kernel_size":2, "stride":2,}),
            Layer("conv3", torch.nn.Conv2d,      256,     None, (), {"kernel_size":2, "stride":2,}),
            Layer("relu3", torch.nn.ReLU,        None,    None, (), {}),
            Layer("pool3", torch.nn.MaxPool2d,   None,    None, (), {"kernel_size":3, "stride":3,}),
            Layer("view6", torch.nn.Identity,    256*1*2, "reshape(x,-1)", (), {}),
            Layer("fc7",   torch.nn.Linear,      256,  None,           (), {}),
            Layer("norm7", torch.nn.BatchNorm1d, 0,    None,           (), {}),
            Layer("relu7", torch.nn.ReLU,        None, None,           (), {}),
            Layer("fc8",   torch.nn.Linear,      128,  None,           (), {}),
            Layer("norm8", torch.nn.BatchNorm1d, 0,    None,           (), {}),
            Layer("relu8", torch.nn.ReLU,        None, None,           (), {}),
            Layer("fc9",   torch.nn.Linear,      64,   None,           (), {}),
            Layer("norm9", torch.nn.BatchNorm1d, 0,    None,           (), {}),
            Layer("relu9", torch.nn.ReLU,        None, None,           (), {}),
            Layer("output",torch.nn.Linear,      len(self.list_action), None, (), {}),
        )
        qfunc = DQN(torch_nn, self.list_action, gamma=gamma, batch_size=128, capacity=1000, lr=0.001)
        qfunc.to_cuda()
        QLearn.__init__(self, qfunc=qfunc, epsilon=epsilon)


    def initialize(self, init_country=None):
        """
        episode単位の初期化
        """
        init_action = self.list_action[0] if init_country is None else init_country
        self.state_now    = self.wmp.conv_to_torch(self.wmp.initialize()) # 地図を初期化
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
        state_prev と action_prev から決まる state を返却する
        """
        action_prev = self.action_prev if action_prev is None else action_prev
        img = self.state_prev.copy()
        img = self.wmp.conv_from_torch(img)
        img = self.wmp.drawline(img, self.action_pprev, action_prev)
        img = self.wmp.drawcities(img)
        img = self.wmp.drawlocation(img, self.action_pprev, (255,255,0))
        img = self.wmp.drawlocation(img, action_prev, (255,255,255))
        img = self.wmp.conv_to_torch(img)
        return img


    def reward(self, state_prev: object=None, action_prev: object=None) -> object:
        """
        state_prev と action_prev から決まる reward を返却する
        """
        state_prev  = self.state_prev  if state_prev  is None else state_prev
        action_prev = self.action_prev if action_prev is None else action_prev
        dist = self.distance(self.action_pprev, action_prev)
        self.loss += dist
        return -self.loss if self.is_finish() else 0


    def q_update(self):
        """
        Q値の更新を行う
        """
        logger.debug(f"state_prev: {self.state_prev}, action_prev: {self.action_prev}, reward_now: {self.reward_now}, state_now: {self.state_now}", color=["YELLOW"])
        self.qfunc.store(self.state_prev, self.action_prev, self.reward_now, self.state_now, prob_actions=self.prob_actions, on_episode=(self.is_finish() == False))
        self.qfunc.update()

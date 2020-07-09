import time, copy
import gym
import torch
import numpy as np
from PIL import Image

# local package
from kkreinforce.lib.kkrl import StateManager
from kkreinforce.lib.qlearn import QLearn, QTable, DQN
from kkreinforce.lib.kknn import TorchNN, Layer
from kkimagemods.util.logger import set_logger
logger = set_logger(__name__)


class CartPole_QTable(QLearn):
    def __init__(self, alpha: float, gamma: float, epsilon: float):
        # CartPole 環境ロード
        self.env = gym.make('CartPole-v0')

        # Action
        self.list_action = np.array([0, 1])

        # State
        self.state_mng: StateManager = StateManager()
        self.state_mng.set_state(0, "numeric_bins", options={"bin_min": -2.4,  "bin_max": 2.4,  "bins": 10}) #カート位置
        self.state_mng.set_state(1, "numeric_bins", options={"bin_min": -3.0,  "bin_max": 3.0,  "bins": 10}) #カート速度
        self.state_mng.set_state(2, "numeric_bins", options={"bin_min": -0.25, "bin_max": 0.25, "bins": 10}) #棒の角度
        self.state_mng.set_state(3, "numeric_bins", options={"bin_min": -2.0,  "bin_max": 2.0,  "bins": 10}) #棒の角速度
        # Action Value Function
        qfunc: QTable = QTable(
            state_list=self.state_mng.pattern(), 
            action_list=np.arange(self.env.env.action_space.n).tolist(), 
            alpha=alpha, gamma=gamma
        )
        QLearn.__init__(self, qfunc=qfunc, epsilon=epsilon)

        # Others
        self.done = False
    
    def initialize(self):
        self.env.reset()
        self.state_now      = tuple(self.state_mng.conv_tmp({i:x for i, x in enumerate(self.env.env.state)}))
        self.state_prev     = self.state_now
        self.action_prev    = self.list_action[0]
        self.reward_now     = None
        self.done           = False
    
    def state(self, state_prev: object=None, action_prev: object=None) -> object:
        """
        state_prevとaction_prevから状態遷移確率に従い、次のstateを決定する
        """
        action_prev = self.action_prev if action_prev is None else action_prev
        observation, _, done, _ = self.env.step(action_prev) # これは何度も実行できない
        self.done = done
        return tuple(self.state_mng.conv_tmp({i:x for i, x in enumerate(observation)}))

    def action_best(self, state_now: object=None) -> object:
        """
        state_now から決定される最適 action を返却する
        """
        state_now = self.state_now if state_now is None else state_now
        _, action = self.qfunc.get_max(state_now, prob_actions=None)
        return action

    def action_random(self) -> object:
        """
        state_now から決定される random action を返却する
        """
        ndf = self.list_action
        return np.random.permutation(ndf)[0]

    def reward(self, state_prev: object=None, action_prev: object=None, state_now: object=None) -> object:
        """
        state_prev と action_prev (と state_now) から決まる reward を返却する
        """
        reward = 1
        if self.is_finish():
            if self.step < 195:
                reward = -200
        return reward

    def is_finish(self) -> bool:
        return self.done

    def q_update(self):
        """
        Q値の更新を行う
        Q(s, a) = Q(s, a) + alpha*(r+gamma*maxQ(s')-Q(s, a))
        """
        logger.debug(f"state_prev: {self.state_prev}, action_prev: {self.action_prev}, reward_now: {self.reward_now}, state_now: {self.state_now}", color=["YELLOW"])
        self.qfunc.update(self.state_prev, self.action_prev, self.reward_now, self.state_now, prob_actions=self.prob_actions, on_episode=(self.is_finish() == False))

    def play(self, save_gif_path: str=None):
        self.init()
        self.is_eval = True
        img_list = []
        while self.is_finish() == False:
            if save_gif_path is not None:
                img = self.env.render(mode="rgb_array")
                img_list.append(Image.fromarray(img))
            self.env.render()
            logger.debug(f"{self.env.env.state}")
            action = self.action_best()
            self.transition(action=action)
            time.sleep(0.05)
        if save_gif_path is not None:
            img_list[0].save(save_gif_path, save_all=True, append_images=img_list[1:], duration=25, loop=0)
        logger.info(f'finish !! all step: {self.step}', color=["BOLD", "GREEN"])



class CartPole_DNN(CartPole_QTable):
    def __init__(self, gamma: float, epsilon: float):
        # CartPole 環境ロード
        self.env = gym.make('CartPole-v0')

        # Action
        self.list_action = np.array([0, 1])

        # Value Function
        nn = TorchNN(
            self.env.env.observation_space.shape[0],
            Layer("fc1",   torch.nn.Linear,      16,   None, (), {}),
            Layer("relu1", torch.nn.ReLU,        None, None, (), {}),
            Layer("fc2",   torch.nn.Linear,      16,   None, (), {}),
            Layer("relu2", torch.nn.ReLU,        None, None, (), {}),
            Layer("fc3",   torch.nn.Linear,      16,   None, (), {}),
            Layer("relu3", torch.nn.ReLU,        None, None, (), {}),
            Layer("fc4",   torch.nn.Linear,      self.env.env.action_space.n, None, (), {}),
        )
        qfunc = DQN(
            nn, self.list_action, 
            gamma=gamma, batch_size=128, capacity=10000, lr=0.0001
        )
        qfunc.to_cuda()
        QLearn.__init__(self, qfunc=qfunc, epsilon=epsilon)

        # Others
        self.done = False
    
    def initialize(self):
        self.state_now   = self.env.reset()
        self.state_prev  = self.state_now
        self.action_prev = self.list_action[0]
        self.reward_now  = None
        self.done        = False
    
    def state(self, state_prev: object=None, action_prev: object=None) -> object:
        """
        state_prevとaction_prevから状態遷移確率に従い、次のstateを決定する
        """
        action_prev = self.action_prev if action_prev is None else action_prev
        observation, _, done, _ = self.env.step(action_prev) # これは何度も実行できない
        self.done = done
        return observation

    def reward(self, state_prev: object=None, action_prev: object=None, state_now: object=None) -> object:
        """
        state_prev と action_prev (と state_now) から決まる reward を返却する
        """
        reward = 0
        if self.is_finish():
            if self.step < 195:
                reward = -1
            else:
                reward = 1
        return reward

    def q_update(self):
        """
        Q値の更新を行う
        Q(s, a) = Q(s, a) + alpha*(r+gamma*maxQ(s')-Q(s, a))
        """
        logger.debug(f"state_prev: {self.state_prev}, action_prev: {self.action_prev}, reward_now: {self.reward_now}, state_now: {self.state_now}", color=["YELLOW"])
        self.qfunc.store(self.state_prev, self.action_prev, self.reward_now, self.state_now, prob_actions=self.prob_actions, on_episode=(self.is_finish() == False))
        self.qfunc.update()

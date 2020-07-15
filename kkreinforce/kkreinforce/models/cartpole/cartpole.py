import time, copy
import gym
import torch
import numpy as np
import cv2
from PIL import Image

# local package
from kkreinforce.lib.kkrl import StateManager
from kkreinforce.lib.qlearn import QLearn, QTable, DQN
from kkreinforce.lib.kknn import TorchNN, Layer
from kkreinforce.lib.policygrad import PolicyGradient, PolicyGradientNN
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

    def train_after_episode(self):
        super().train_after_episode()
        logger.info(f"step: {self.step}")

    def play(self, display: bool = False, save_gif_path: str=None):
        self.init()
        self.is_eval = True
        img_list = []
        while self.is_finish() == False:
            if display and save_gif_path is not None:
                img = self.env.render(mode="rgb_array")
                img_list.append(Image.fromarray(img.copy()))
            if display:
                self.env.render()
            logger.debug(f"{self.env.env.state}")
            action = self.action_best()
            self.transition(action=action)
            time.sleep(0.05)
        if display and save_gif_path is not None:
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
        self.qfunc.update()
    
    def train_after_step(self):
        self.qfunc.store(self.state_prev, self.action_prev, self.reward_now, self.state_now, prob_actions=self.prob_actions, on_episode=(self.is_finish() == False))

    def train_after_episode(self):
        self.q_update()



class CartPole_CNN(CartPole_DNN):
    def __init__(self, gamma: float, epsilon: float):
        # CartPole 環境ロード
        self.env = gym.make('CartPole-v0')

        # Action
        self.list_action = np.array([0, 1])

        # Value Function
        nn = TorchNN(3,
            Layer("conv1", torch.nn.Conv2d,      128,   None, (), {"kernel_size":3, "stride":1,}),
            Layer("relu1", torch.nn.ReLU,        None,  None, (), {}),
            Layer("pool1", torch.nn.MaxPool2d,   None,  None, (), {"kernel_size":2, "stride":2,}),
            Layer("conv2", torch.nn.Conv2d,      128,   None, (), {"kernel_size":2, "stride":1,}),
            Layer("relu2", torch.nn.ReLU,        None,  None, (), {}),
            Layer("pool2", torch.nn.MaxPool2d,   None,  None, (), {"kernel_size":2, "stride":2,}),
            Layer("conv3", torch.nn.Conv2d,      64,    None, (), {"kernel_size":2, "stride":2,}),
            Layer("relu3", torch.nn.ReLU,        None,  None, (), {}),
            Layer("pool3", torch.nn.MaxPool2d,   None,  None, (), {"kernel_size":2, "stride":2,}),
            Layer("conv4", torch.nn.Conv2d,      64,    None, (), {"kernel_size":2, "stride":2,}),
            Layer("relu4", torch.nn.ReLU,        None,  None, (), {}),
            Layer("pool4", torch.nn.MaxPool2d,   None,  None, (), {"kernel_size":2, "stride":2,}),
            Layer("conv5", torch.nn.Conv2d,      32,    None, (), {"kernel_size":2, "stride":1,}),
            Layer("relu5", torch.nn.ReLU,        None,  None, (), {}),
            Layer("pool5", torch.nn.MaxPool2d,   None,  None, (), {"kernel_size":2, "stride":2,}),
            Layer("view6", torch.nn.Identity,    32*1*4,"reshape(x,-1)", (), {}),
            Layer("fc6",   torch.nn.Linear,      64,    None,           (), {}),
            Layer("norm6", torch.nn.BatchNorm1d, 0,     None,           (), {}),
            Layer("relu6", torch.nn.ReLU,        None,  None,           (), {}),
            Layer("fc7",   torch.nn.Linear,      32,    None,           (), {}),
            Layer("norm7", torch.nn.BatchNorm1d, 0,     None,           (), {}),
            Layer("relu7", torch.nn.ReLU,        None,  None,           (), {}),
            Layer("output",torch.nn.Linear,      self.env.env.action_space.n, None, (), {}),
        )
        qfunc = DQN(
            nn, self.list_action, 
            gamma=gamma, batch_size=16, capacity=500, lr=0.0001
        )
        qfunc.to_cuda()
        QLearn.__init__(self, qfunc=qfunc, epsilon=epsilon)

        # Others
        self.done = False
    
    @classmethod
    def conv_to_torch(cls, img: np.ndarray) -> np.ndarray:
        return np.array([img[:, :, i] for i in range(img.shape[-1])])
    
    @classmethod
    def conv_from_torch(cls, img: np.ndarray) -> np.ndarray:
        return cv2.flip(np.rot90(img.T, 3), 1)

    def initialize(self):
        self.env.reset()
        img = self.env.render(mode="rgb_array")[150:-50,:]
        self.state_now   = self.conv_to_torch(img)
        self.state_prev  = self.state_now
        self.action_prev = self.list_action[0]
        self.reward_now  = None
        self.done        = False
    
    def state(self, state_prev: object=None, action_prev: object=None) -> object:
        """
        state_prevとaction_prevから状態遷移確率に従い、次のstateを決定する
        """
        action_prev = self.action_prev if action_prev is None else action_prev
        _, _, done, _ = self.env.step(action_prev) # これは何度も実行できない
        self.done = done
        img = self.env.render(mode="rgb_array")[150:-50,:]
        return self.conv_to_torch(img)



class CartPole_CNN2(CartPole_CNN):
    def __init__(self, gamma: float, epsilon: float):
        # CartPole 環境ロード
        self.env = gym.make('CartPole-v0')

        # Action
        self.list_action = np.array([0, 1])

        # Value Function
        cnn1 = TorchNN(3,
            Layer("conv1", torch.nn.Conv2d,      128,   None, (), {"kernel_size":3, "stride":1,}),
            Layer("relu1", torch.nn.ReLU,        None,  None, (), {}),
            Layer("pool1", torch.nn.MaxPool2d,   None,  None, (), {"kernel_size":2, "stride":2,}),
            Layer("conv2", torch.nn.Conv2d,      128,   None, (), {"kernel_size":2, "stride":1,}),
            Layer("relu2", torch.nn.ReLU,        None,  None, (), {}),
            Layer("pool2", torch.nn.MaxPool2d,   None,  None, (), {"kernel_size":2, "stride":2,}),
            Layer("conv3", torch.nn.Conv2d,      64,    None, (), {"kernel_size":2, "stride":2,}),
            Layer("relu3", torch.nn.ReLU,        None,  None, (), {}),
            Layer("pool3", torch.nn.MaxPool2d,   None,  None, (), {"kernel_size":2, "stride":2,}),
            Layer("conv4", torch.nn.Conv2d,      64,    None, (), {"kernel_size":2, "stride":2,}),
            Layer("relu4", torch.nn.ReLU,        None,  None, (), {}),
            Layer("pool4", torch.nn.MaxPool2d,   None,  None, (), {"kernel_size":2, "stride":2,}),
            Layer("conv5", torch.nn.Conv2d,      32,    None, (), {"kernel_size":2, "stride":1,}),
            Layer("relu5", torch.nn.ReLU,        None,  None, (), {}),
            Layer("pool5", torch.nn.MaxPool2d,   None,  None, (), {"kernel_size":2, "stride":2,}),
        )
        cnn2 = TorchNN(3,
            Layer("conv1", torch.nn.Conv2d,      128,   None, (), {"kernel_size":3, "stride":1,}),
            Layer("relu1", torch.nn.ReLU,        None,  None, (), {}),
            Layer("pool1", torch.nn.MaxPool2d,   None,  None, (), {"kernel_size":2, "stride":2,}),
            Layer("conv2", torch.nn.Conv2d,      128,   None, (), {"kernel_size":2, "stride":1,}),
            Layer("relu2", torch.nn.ReLU,        None,  None, (), {}),
            Layer("pool2", torch.nn.MaxPool2d,   None,  None, (), {"kernel_size":2, "stride":2,}),
            Layer("conv3", torch.nn.Conv2d,      64,    None, (), {"kernel_size":2, "stride":2,}),
            Layer("relu3", torch.nn.ReLU,        None,  None, (), {}),
            Layer("pool3", torch.nn.MaxPool2d,   None,  None, (), {"kernel_size":2, "stride":2,}),
            Layer("conv4", torch.nn.Conv2d,      64,    None, (), {"kernel_size":2, "stride":2,}),
            Layer("relu4", torch.nn.ReLU,        None,  None, (), {}),
            Layer("pool4", torch.nn.MaxPool2d,   None,  None, (), {"kernel_size":2, "stride":2,}),
            Layer("conv5", torch.nn.Conv2d,      32,    None, (), {"kernel_size":2, "stride":1,}),
            Layer("relu5", torch.nn.ReLU,        None,  None, (), {}),
            Layer("pool5", torch.nn.MaxPool2d,   None,  None, (), {"kernel_size":2, "stride":2,}),
        )
        cnn3 = TorchNN(3,
            Layer("conv1", torch.nn.Conv2d,      128,   None, (), {"kernel_size":3, "stride":1,}),
            Layer("relu1", torch.nn.ReLU,        None,  None, (), {}),
            Layer("pool1", torch.nn.MaxPool2d,   None,  None, (), {"kernel_size":2, "stride":2,}),
            Layer("conv2", torch.nn.Conv2d,      128,   None, (), {"kernel_size":2, "stride":1,}),
            Layer("relu2", torch.nn.ReLU,        None,  None, (), {}),
            Layer("pool2", torch.nn.MaxPool2d,   None,  None, (), {"kernel_size":2, "stride":2,}),
            Layer("conv3", torch.nn.Conv2d,      64,    None, (), {"kernel_size":2, "stride":2,}),
            Layer("relu3", torch.nn.ReLU,        None,  None, (), {}),
            Layer("pool3", torch.nn.MaxPool2d,   None,  None, (), {"kernel_size":2, "stride":2,}),
            Layer("conv4", torch.nn.Conv2d,      64,    None, (), {"kernel_size":2, "stride":2,}),
            Layer("relu4", torch.nn.ReLU,        None,  None, (), {}),
            Layer("pool4", torch.nn.MaxPool2d,   None,  None, (), {"kernel_size":2, "stride":2,}),
            Layer("conv5", torch.nn.Conv2d,      32,    None, (), {"kernel_size":2, "stride":1,}),
            Layer("relu5", torch.nn.ReLU,        None,  None, (), {}),
            Layer("pool5", torch.nn.MaxPool2d,   None,  None, (), {"kernel_size":2, "stride":2,}),
        )
        nn = TorchNN(None,
            Layer("split", torch.nn.Identity,    None,     "split_numpy_3", (), {}),
            Layer("cnn1",  cnn1,                 None,     None,    (), {}),
            Layer("cnn2",  cnn2,                 None,     None,    (), {}),
            Layer("cnn3",  cnn3,                 None,     None,    (), {}),
            Layer("view6", torch.nn.Identity,    3*32*1*4, "combine", (), {}),
            Layer("fc6",   torch.nn.Linear,      128,      None,           (), {}),
            Layer("norm6", torch.nn.BatchNorm1d, 0,        None,           (), {}),
            Layer("relu6", torch.nn.ReLU,        None,     None,           (), {}),
            Layer("fc7",   torch.nn.Linear,      64,       None,           (), {}),
            Layer("norm7", torch.nn.BatchNorm1d, 0,        None,           (), {}),
            Layer("relu7", torch.nn.ReLU,        None,     None,           (), {}),
            Layer("output",torch.nn.Linear,      self.env.env.action_space.n, None, (), {}),
        )
        qfunc = DQN(
            nn, self.list_action, 
            gamma=gamma, batch_size=8, capacity=256, lr=0.0001
        )
        qfunc.to_cuda()
        QLearn.__init__(self, qfunc=qfunc, epsilon=epsilon)

        # Others
        self.done = False
        self.history = [None, None, None, None, None]

    def initialize(self):
        self.env.reset()
        img = self.env.render(mode="rgb_array")[150:-50,:]
        img = self.conv_to_torch(img)
        self.history     = [img.copy(), img.copy(), img.copy(), img.copy(), img.copy()]
        self.state_now   = np.concatenate([img.reshape(1, *img.shape), img.reshape(1, *img.shape), img.reshape(1, *img.shape)], axis=0)
        self.state_prev  = self.state_now
        self.action_prev = self.list_action[0]
        self.reward_now  = None
        self.done        = False
    
    def state(self, state_prev: object=None, action_prev: object=None) -> object:
        """
        state_prevとaction_prevから状態遷移確率に従い、次のstateを決定する
        """
        action_prev = self.action_prev if action_prev is None else action_prev
        _, _, done, _ = self.env.step(action_prev) # これは何度も実行できない
        self.done = done
        img = self.conv_to_torch(self.env.render(mode="rgb_array")[150:-50,:])
        self.history.append(img)
        del self.history[0]
        img = np.concatenate([
            img.reshape(1, *img.shape), 
            self.history[2].reshape(1, *self.history[2].shape),
            self.history[0].reshape(1, *self.history[0].shape),
            ], axis=0)
        return img



class CartPole_CNN3(CartPole_CNN):
    def __init__(self, gamma: float, epsilon: float):
        # CartPole 環境ロード
        self.env = gym.make('CartPole-v0')

        # Action
        self.list_action = np.array([0, 1])

        # Value Function
        self.insize = 20
        nn = TorchNN(self.insize,
            Layer("conv1",  torch.nn.Conv2d,      64,   None, (), {"kernel_size":20, "stride":10,}),
            Layer("relu1",  torch.nn.ReLU,        None,  None, (), {}),
            Layer("conv2",  torch.nn.Conv2d,      64,   None, (), {"kernel_size":5, "stride":5}),
            Layer("relu2",  torch.nn.ReLU,        None,  None, (), {}),
            Layer("view7",  torch.nn.Identity,    64*3*11,"reshape(x,-1)", (), {}),
            Layer("fc7",    torch.nn.Linear,      512,   None,           (), {}),
            Layer("norm7",  torch.nn.BatchNorm1d, 0,     None,           (), {}),
            Layer("relu7",  torch.nn.ReLU,        None,  None,           (), {}),
            Layer("fc8",    torch.nn.Linear,      256,   None,           (), {}),
            Layer("norm8",  torch.nn.BatchNorm1d, 0,     None,           (), {}),
            Layer("relu8",  torch.nn.ReLU,        None,  None,           (), {}),
            Layer("fc9",    torch.nn.Linear,      64,    None,           (), {}),
            Layer("norm9",  torch.nn.BatchNorm1d, 0,     None,           (), {}),
            Layer("relu9",  torch.nn.ReLU,        None,  None,           (), {}),
            Layer("output", torch.nn.Linear,      self.env.env.action_space.n, None, (), {}),
        )
        qfunc = DQN(
            nn, self.list_action, 
            gamma=gamma, batch_size=192, capacity=1000, lr=0.001
        )
        qfunc.to_cuda()
        QLearn.__init__(self, qfunc=qfunc, epsilon=epsilon)

        # Others
        self.done = False
        self.history = [None for _ in range(self.insize)]

    def initialize(self):
        self.env.reset()
        img = self.env.render(mode="rgb_array")[150:-50,:]
        img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
        img[img <  255] = 1 # 白以外の場所を1に
        img[img == 255] = 0 # 白の場所を0にする
        self.history     = [np.zeros_like(img) for _ in range(self.insize)]
        self.history.append(img.copy())
        del self.history[0]
        img = np.concatenate([self.history[-i].reshape(1, *self.history[-i].shape) for i in range(1,self.insize+1)], axis=0)
        self.state_now   = img
        self.state_prev  = self.state_now
        self.action_prev = self.list_action[0]
        self.reward_now  = None
        self.done        = False
    
    def state(self, state_prev: object=None, action_prev: object=None) -> object:
        """
        state_prevとaction_prevから状態遷移確率に従い、次のstateを決定する
        """
        action_prev = self.action_prev if action_prev is None else action_prev
        state_prev  = self.state_prev  if state_prev  is None else state_prev
        _, _, done, _ = self.env.step(action_prev) # これは何度も実行できない
        self.done = done
        img = self.env.render(mode="rgb_array")[150:-50,:]
        img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
        img[img <  255] = 1 # 白以外の場所を1に
        img[img == 255] = 0 # 白の場所を0にする
        self.history.append(img.copy())
        del self.history[0]
        img = np.concatenate([self.history[-i].reshape(1, *self.history[-i].shape) for i in range(1,self.insize+1)], axis=0)
        return img



class CartPole_PG(PolicyGradient):
    def __init__(self, gamma: float):
        # CartPole 環境ロード
        self.env = gym.make('CartPole-v1')

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
            Layer("output",torch.nn.Softmax,     None, None, (), {"dim":1}),
        )
        policy = PolicyGradientNN(
            nn, self.list_action, batch_size=-1, lr=0.001
        )
        policy.to_cuda()
        super().__init__(policy=policy, list_action=self.list_action, gamma=gamma)

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
        reward = 1
        if self.is_finish():
            if self.step < 480:
                reward = -500
        return reward

    def is_finish(self) -> bool:
        return self.done

    def train_after_episode(self):
        super().train_after_episode()
        logger.info(f"step: {self.step}")

    def play(self, display: bool = False, save_gif_path: str=None):
        self.init()
        self.is_eval = True
        img_list = []
        while self.is_finish() == False:
            if display and save_gif_path is not None:
                img = self.env.render(mode="rgb_array")
                img_list.append(Image.fromarray(img))
            if display:
                self.env.render()
            logger.debug(f"{self.env.env.state}")
            action = self.action()
            self.transition(action=action)
            time.sleep(0.05)
        if display and save_gif_path is not None:
            img_list[0].save(save_gif_path, save_all=True, append_images=img_list[1:], duration=25, loop=0)
        logger.info(f'finish !! all step: {self.step}', color=["BOLD", "GREEN"])

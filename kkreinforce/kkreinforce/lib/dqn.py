import copy
from typing import List, Tuple
import torch
from torch import nn
import pandas as pd
import numpy as np
from collections import namedtuple
import random

# local package
from kkimagemods.util.logger import set_logger, set_loglevel
_logname = __name__
logger = set_logger(__name__)





if __name__ == "__main__":
    """ Debug code """
    list_state  = ["aaa", "bbb", "ccc", "ddd"]
    list_action = [1, 2, 3, 4]
    torch_nn = TorchNN(
        len(list_action),
        Layer("lstm",  torch.nn.LSTM,   128,   "rnn_all", (), {}),
        Layer("relu1", torch.nn.ReLU,   None,  None, (), {}),
        Layer("fc2",   torch.nn.Linear, 128,   None, (), {}),
        Layer("relu2", torch.nn.ReLU,   None,  None, (), {}),
        Layer("fc3",   torch.nn.Linear, len(list_action), None, (), {}),
    )
    qtable = DQN(torch_nn, list_state, list_action, alpha=0.5, gamma=0.5, batch_size=128, capacity=200, unit_memory="episode")
    for i in range(100):
        x = np.random.permutation(np.arange(len(list_state)))
        y = np.random.permutation(np.arange(len(list_action)))
        qtable.update(list_state[x[0]], list_action[y[0]], 1, list_state[x[1]], on_episode=True)
        if i % 10 == 0:
            qtable.update(list_state[x[0]], list_action[y[0]], 1, list_state[x[1]], on_episode=False)
    state, action, reward, state_next = qtable.memory.sample(4)
    state = qtable.conv_onehot(state)
    
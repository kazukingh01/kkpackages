import numpy as np
from functools import partial
from scipy.misc import derivative

# local packages
from kkutils.util.com import set_logger
logger = set_logger(__name__)


__all__ = [
    "softmax",
    "sigmoid",
    "rmse",
    "mae",
    "binary_cross_entropy",
    "multi_cross_entropy",
    "huber_loss",
    "huber_loss_grad",
    "focal_loss",
    "focal_loss_grad",
    "calc_grad_hess",
    "func_embed",
]


def softmax(x):
    f = np.exp(x)/np.sum(np.exp(x), axis = 1, keepdims = True)
    return f

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def rmse(x: np.ndarray, t: np.ndarray):
    return (x - t) ** 2 / 2.

def mae(x: np.ndarray, t: np.ndarray):
    return np.abs(x - t)

def binary_cross_entropy(x: np.ndarray, t: np.ndarray):
    t = t.astype(np.int32)
    x = sigmoid(x)
    return -1 * (t * np.log(x) + (1 - t) * np.log(1 - x))

def multi_cross_entropy(x: np.ndarray, t: np.ndarray, is_standardize=False):
    t = t.astype(np.int32)
    if len(x.shape) > 1:
        if is_standardize:
            x = softmax(x) # softmax で確率化
        t = np.identity(x.shape[1])[t]
        return -1 * t * np.log(np.clip(x, 1e-10, 1))
    else:
        if is_standardize:
            x = sigmoid(x)
        x[t == 0] = 1 - x[t == 0] # 0ラベル箇所は確率を反転する
        return -1 * np.log(np.clip(x, 1e-10, 1))

def huber_loss(x: np.ndarray, t: np.ndarray, delta: float=1.0):
    loss = np.abs(t - x)
    mask = (loss <= delta)
    loss[ mask] = (loss[mask] ** 2) / 2.
    loss[~mask] = delta * loss[~mask] - (delta ** 2) / 2.
    return loss

def huber_loss_grad(x: np.ndarray, t: np.ndarray, delta: float=1.0):
    mask = (np.abs(t - x) <= delta)
    grad = np.zeros_like(x).astype(np.float32)
    grad[ mask] = x[mask] - t[mask]
    grad[~mask] = delta # 先に全部をdelta
    grad[((x - t) < (-1 * delta))] = -delta # delta未満を-delta
    hess = np.zeros_like(x).astype(np.float32)
    hess[ mask] = 1.0
    hess[~mask] = 0.0
    return grad, hess

def focal_loss(x: np.ndarray, t: np.ndarray, gamma: float=1) -> np.ndarray:
    """
    Params::
        x:
            予測値. 規格化されていない値であること. 1次元
            >>> x
            array([0.65634349, 0.8510698 , 0.61597224])
            multi classのとき. 例えば 3 class の場合は下記のような値になっていること.
            >>> x
            array([
                [0.65634349, 0.8510698 , 0.61597224],
                [0.58012161, 0.79659195, 0.39168051]
            ])
        t:
            正解値.
            multi classのとき. 例えば 3 class の場合は [0,0,0,1,1,1,2,1,2,0,0,...](0〜2まで)
        multi class に対してどうするか。categorycal cross entropy と同じ考え方で、正解ラベル以外の列はlossが0になる
        ※このアルゴリズムテクニックだけ残しておく。今は使っていない
        ※x[np.arange(t.shape[0]).reshape(-1, 1), t.reshape(-1, 1)] = 1 - x[np.arange(t.shape[0]).reshape(-1, 1), t.reshape(-1, 1)]
    """
    t = t.astype(np.int32)
    if len(x.shape) > 1:
        x = softmax(x) # softmax で確率化
        t = np.identity(x.shape[1])[t]
        return -1 * t * (1 - x)**gamma * np.log(x)
    else:
        x = sigmoid(x)
        x[t == 0] = 1 - x[t == 0] # 0ラベル箇所は確率を反転する
        return -1 * (1 - x)**gamma * np.log(x)

def focal_loss_grad(x: np.ndarray, t: np.ndarray, gamma: float=1) -> (np.ndarray, np.ndarray, ):
    """
    内部に softmax を含む関数については derivative では計算が安定しない.
    かなり面倒だが、真面目に微分してみる。参考 https://hackmd.io/OddWU6zlR2GkrZNsl4IPnA
    """
    t = t.astype(np.int32)
    if len(x.shape) > 1:
        x = softmax(x) # softmax で確率化
        # 正解列を抜き出し
        xK = x[np.arange(t.shape[0]).reshape(-1, 1), t.reshape(-1, 1)]
        xK = np.tile(xK, (1, x.shape[1]))
        # x1 は 不正解列に -1 をかけて、さらに正解列はそこから1を足す操作
        x1 = x.copy()
        x1 = -1 * x1
        x1[np.arange(t.shape[0]).reshape(-1, 1), t.reshape(-1, 1)] = x1[np.arange(t.shape[0]).reshape(-1, 1), t.reshape(-1, 1)] + 1
        dfdy = gamma * (1 - xK) ** (gamma-1) * np.log(xK) - ((1 - xK) ** gamma / xK)
        dydx = xK * x1
        grad = dfdy * dydx

        dfdydx = dydx * (2 * gamma * (1 - xK) ** (gamma - 1) / xK - gamma * (gamma - 1) * np.log(xK) * (1 - xK) ** (gamma - 2) + (1 - xK) ** gamma * (xK ** -2))
        dydxdx = dydx * (1 - 2 * x)
        hess = dfdy * dydxdx + dydx * dfdydx
    else:
        grad = derivative(lambda _x: focal_loss(_x, t, gamma=gamma), x, n=1, dx=1e-6)
        hess = derivative(lambda _x: focal_loss(_x, t, gamma=gamma), x, n=2, dx=1e-6)
    return grad, hess

def calc_grad_hess(x: np.ndarray, t: np.ndarray, loss_func=None, dx=1e-6, **kwargs) -> (np.ndarray, np.ndarray, ):
    logger.debug(f'dx: {dx}, loss: {loss_func(x, t, **kwargs)}')
    grad = derivative(lambda _x: loss_func(_x, t, **kwargs), x, n=1, dx=dx)
    hess = derivative(lambda _x: loss_func(_x, t, **kwargs), x, n=2, dx=dx)
    return grad, hess

def func_embed(func: str, calc_type: str=None, **kwargs):
    def _work(x: np.ndarray, t: np.ndarray, func=None, calc_type: str=None, **kwargs):
        if isinstance(calc_type, str):
            if   calc_type == "mean":
                return np.mean(func(x, t, **kwargs))
            elif calc_type == "sum":
                return np.sum(func(x, t, **kwargs))
        else:
            return func(x, t, **kwargs)
    if isinstance(func, str):
        if   func in ["mae"]:
            return partial(_work, func=mae, calc_type=calc_type, **kwargs)
        elif func in ["rmse"]:
            return partial(_work, func=rmse, calc_type=calc_type, **kwargs)
        elif func in ["bce"]:
            return partial(_work, func=binary_cross_entropy, calc_type=calc_type, **kwargs)
        elif func in ["mce"]:
            return partial(_work, func=multi_cross_entropy, calc_type=calc_type, **kwargs)
    else:
        return partial(_work, func=func, calc_type=calc_type, **kwargs)
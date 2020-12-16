import numpy as np
import lightgbm as lgb

# local package
from kkutils.util.com import is_callable


__all__ = [
    "lgb_custom_objective",
    "lgb_custom_eval"
]

def lgb_custom_objective(y_pred: np.ndarray, data: lgb.Dataset, func_loss, is_lgbdataset: bool=True):
    """
    lightGBMのcustomized objectiveの共通関数
    Params::
        y_pred:
            予測値. multi classの場合は、n_sample * n_class の長さになったいる
            値は、array([0データ目0ラベルの予測値, ..., Nデータ目0ラベルの予測値, 0データ目1ラベルの予測値, ..., ])
        data:
            train_set に set した値
        func_loss:
            y_pred, y_true を入力に持ち、y_pred と同じ shape を持つ return をする
        is_lgbdataset:
            lgb.dataset でなかった場合は入力が逆転するので気をつける
    """
    if is_lgbdataset == False:
        y_true = y_pred.copy()
        y_pred = data
    else:
        y_true = data.label
        if is_callable(data, "ndf_label"):
            y_true = data.get_culstom_label(y_true.astype(int))
    if y_pred.shape[0] != y_true.shape[0]:
        # multi class の場合
        y_pred = y_pred.reshape(-1 , y_true.shape[0]).T
    grad, hess = func_loss(y_pred, y_true)
    return grad.T.reshape(-1), hess.T.reshape(-1)

def lgb_custom_eval(y_pred: np.ndarray, data: lgb.Dataset, func_loss, func_name: str, is_higher_better: bool, is_lgbdataset: bool=True):
    """
    lightGBMのcustomized objectiveの共通関数
    Params::
        y_pred:
            予測値. multi classの場合は、n_sample * n_class の長さになったいる
            値は、array([0データ目0ラベルの予測値, ..., Nデータ目0ラベルの予測値, 0データ目1ラベルの予測値, ..., ])
        data:
            train_set に set した値
        func_loss:
            y_pred, y_true を入力に持ち、grad, hess を return する関数
    """
    if is_lgbdataset == False:
        y_true = y_pred.copy()
        y_pred = data
    else:
        y_true = data.label
        if is_callable(data, "ndf_label"):
            y_true = data.get_culstom_label(y_true.astype(int))
    if y_pred.shape[0] != y_true.shape[0]:
        # multi class の場合
        y_pred = y_pred.reshape(-1 , y_true.shape[0]).T
    value = func_loss(y_pred, y_true)
    return func_name, np.mean(value), is_higher_better

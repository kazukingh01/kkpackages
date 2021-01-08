import datetime
import pandas as pd
import numpy as np
from sklearn.exceptions import NotFittedError
from typing import List, Callable
from functools import partial
import lightgbm as lgb
from lightgbm import LGBMClassifier, LGBMRegressor, LGBMModel, Dataset
from lightgbm.callback import record_evaluation, EarlyStopException, _format_eval_result
from optuna.integration import lightgbm as lgbtune

# local package
from kkutils.util.ml import lgb_custom_eval, lgb_custom_objective
from kkutils.util.numpy import calc_grad_hess, func_embed
from kkutils.util.com import set_logger, MyLogger
logger = set_logger(__name__)


__all__ = [
    "KkLGBMModelBase",
    "KkLGBMClassifier",
    "KkLGBMRegressor",
    "train",
    "autotuner",
    "tune",
]


class KkLGBMModelBase(LGBMModel):
    """
    LGBMClassifier, LGBMRegressorの共通化処理のBaseClass
    """
    def _fit(self, X, y, *argv, **kwargs):
        logger.info("START")
        self.dict_eval      = {}
        self.dict_eval_best = {}
        self.dict_eval_hist = []
        self.save_interval  = kwargs.get("save_interval")
        if kwargs.get("eval_set") is None:
            kwargs["eval_set"] = [(X, y, ),]
        else:
            if   isinstance(kwargs.get("eval_set"), tuple):
                kwargs["eval_set"] = [(X, y, ), kwargs["eval_set"]]
            elif isinstance(kwargs.get("eval_set"), list):
                kwargs["eval_set"].insert(0, (X, y, ))
        if kwargs.get("eval_names") is None:
            kwargs["eval_names"] = []
            for i in range(len(kwargs["eval_set"])):
                if i == 0: kwargs["eval_names"].append("train")
                else:      kwargs["eval_names"].append(f"valid_{i}")
        else:
            if isinstance(kwargs.get("eval_names"), list):
                kwargs["eval_names"].insert(0, "train")
        if kwargs.get("callbacks") is None:
            kwargs["callbacks"] = [
                record_evaluation(self.dict_eval),
                self.print_evaluation(logger),
                self.callback_best_iter(self.dict_eval_best, kwargs.get("early_stopping_rounds"), logger)
            ]
            if self.save_interval is not None: kwargs["callbacks"].append(self.callback_model_save(self.save_interval))
        # model type によって fit の先を変える
        self.fit_common(X, y, *argv, **kwargs)
        self.dict_eval_hist.append(self.dict_eval.copy())
        try:
            if isinstance(self.best_iteration_, int) and self.save_interval is not None:
                base_step            = self.best_iteration_ - (self.best_iteration_ // self.save_interval)
                self.n_estimators    = base_step + 100
                kwargs["init_model"] = f"./model_{base_step}.txt"
                kwargs["callbacks"]  = [
                    record_evaluation(self.dict_eval), 
                    self.print_evaluation(logger),
                    self.callback_best_iter(self.dict_eval_best, kwargs.get("early_stopping_rounds"), logger),
                    self.callback_lr_schedule([base_step], lr_decay=0.2)
                ]
                logger.info(f'best model is {kwargs["init_model"]}. base_step: {base_step}, re-fitting: \n{self}')
                self.fit_common(X, y, *argv, **kwargs)
        except NotFittedError:
            # Fitting されていない場合に best_iteration_ にアクセスしたらこのエラーが発生する
            pass
        logger.info("END")
    

    def fit_common(self, X, y, *argv, **kwargs):
        raise NotImplementedError
    

    @classmethod
    def callback_model_save(cls, save_interval: int):
        def _callback(env):
            if (env.iteration % save_interval) == 0:
                env.model.save_model(f'model_{env.iteration}.txt')
        _callback.order = 100
        return _callback


    @classmethod
    def callback_best_iter(cls, dict_eval: dict, stopping_rounds: int, logger: MyLogger):
        """
        valid_1 に対してのbest iterationを決める
        """
        def _init(env):
            dict_eval["best_iter"]  = 0
            dict_eval["eval_name"]  = ""
            dict_eval["best_score"] = float("inf")
            dict_eval["best_result_list"] = []
        def _callback(env):
            if not dict_eval:
                _init(env)
            for data_name, eval_name, result, _ in env.evaluation_result_list:
                if data_name == "valid_1":
                    if dict_eval["best_score"] > result:
                        dict_eval["best_score"] = result
                        dict_eval["eval_name"]  = eval_name
                        dict_eval["best_iter"]  = env.iteration
                        dict_eval["best_result_list"] = env.evaluation_result_list
                    break
            if isinstance(stopping_rounds, int) and env.iteration - dict_eval["best_iter"] >= stopping_rounds:
                logger.info(f'early stopping. iteration: {dict_eval["best_iter"]}, score: {dict_eval["best_score"]}')
                raise EarlyStopException(dict_eval["best_iter"], dict_eval["best_result_list"])
        _callback.order = 200
        return _callback


    @classmethod
    def callback_lr_schedule(cls, lr_steps: List[int], lr_decay: float=0.2):
        def _callback(env):
            if int(env.iteration - env.begin_iteration) in lr_steps:
                lr = env.params.get("learning_rate", None)
                dictwk = {"learning_rate": lr * lr_decay}
                env.model.reset_parameter(dictwk)
                env.params.update(dictwk)
        _callback.before_iteration = True
        _callback.order = 100
        return _callback


    @classmethod
    def print_evaluation(cls, logger: MyLogger, period=1, show_stdv=True):
        def _callback(env):
            if period > 0 and env.evaluation_result_list and (env.iteration + 1) % period == 0:
                result = '\t'.join([_format_eval_result(x, show_stdv) for x in env.evaluation_result_list])
                logger.info('[%d]\t%s' % (env.iteration + 1, result))
        _callback.order = 10
        return _callback


    def rm_objective(self):
        """
        lambda x: で定義した function は消さないと pickle 化できない
        """
        self.objective  = None
        self._objective = None
        self._fobj      = None


class KkLGBMClassifier(LGBMClassifier, KkLGBMModelBase):
    def fit(self, X, y, *argv, **kwargs):
        super()._fit(X, y, *argv, **kwargs)
    def fit_common(self, X, y, *argv, **kwargs):
        super().fit(X, y, *argv, **kwargs)


class KkLGBMRegressor(LGBMRegressor, KkLGBMModelBase):
    def fit(self, X, y, *argv, **kwargs):
        super()._fit(X, y, *argv, **kwargs)
    def fit_common(self, X, y, *argv, **kwargs):
        super().fit(X, y, *argv, **kwargs)


class KkLgbDataset(Dataset):
    """
    Usage::
        >>> dataset = KkLgbDataset(x_train)
        >>> dataset.set_culstom_label(y_train)
    """
    def set_culstom_label(self, label: np.ndarray):
        self.label     = np.arange(label.shape[0]).astype(int)
        self.ndf_label = label
    def get_culstom_label(self, indexes: np.ndarray) -> np.ndarray:
        return self.ndf_label[indexes]


def _train(
    params: dict, x_train: np.ndarray, y_train: np.ndarray, loss_func, *args, 
    x_valid: np.ndarray=None, y_valid: np.ndarray=None, 
    loss_func_grad: Callable[[float, float], float]=None, 
    loss_func_eval: Callable[[float, float], float]=None, 
    func_train=None,
    **kwargs
):
    logger.info("START")
    dataset = KkLgbDataset(x_train)
    dataset.set_culstom_label(y_train)
    if not (isinstance(x_valid, list) or isinstance(x_valid, tuple)):
        x_valid = [] if x_valid is None else [x_valid]
        y_valid = [] if y_valid is None else [y_valid]
    list_dataset_valid = [dataset]
    for _x_valid, _y_valid in zip(x_valid, y_valid):
        list_dataset_valid.append(KkLgbDataset(_x_valid))
        list_dataset_valid[-1].set_culstom_label(_y_valid)
    fobj = None
    if loss_func_grad is None and (not isinstance(loss_func, str)):
        loss_func_grad = partial(calc_grad_hess, loss_func=loss_func)
    if loss_func_grad is not None:
        fobj = lambda x,y: lgb_custom_objective(x, y, loss_func_grad, is_lgbdataset=True)
    feval = None
    if loss_func_eval is not None and (not isinstance(loss_func_eval, str)):
        feval = lambda x,y: lgb_custom_eval(x, y, func_embed(loss_func_eval, calc_type="mean"), "myloss", is_higher_better=False, is_lgbdataset=True)
    elif loss_func_eval is None:
        feval = lambda x,y: lgb_custom_eval(x, y, func_embed(loss_func,      calc_type="mean"), "myloss", is_higher_better=False, is_lgbdataset=True)
    if fobj  is None and isinstance(loss_func,      str): params["objective"] = loss_func
    if feval is None and isinstance(loss_func_eval, str): params["metric"]    = loss_func_eval
    evals_result = {} # metric の履歴
    obj = func_train(
        params, dataset, 
        valid_sets=list_dataset_valid, valid_names=["train"]+["valid"+str(i) for i in range(len(list_dataset_valid)-1)],
        fobj =fobj, feval=feval, evals_result=evals_result,
        **kwargs
    )
    logger.info("END")
    return obj


def train(
    params, x_train, y_train, loss_func: str, *args, 
    x_valid: np.ndarray=None, y_valid: np.ndarray=None, 
    loss_func_grad: Callable[[float, float], float]=None, 
    loss_func_eval: Callable[[float, float], float]=None, 
    **kwargs
):
    logger.info("START")
    obj = _train(
        params, x_train, y_train, loss_func, *args, 
        x_valid=x_valid, y_valid=y_valid,
        loss_func_grad=loss_func_grad,
        loss_func_eval=loss_func_eval,
        func_train=lgb.train,
        **kwargs
    )
    logger.info("END")
    return obj


def autotuner(
    params: dict, x_train: np.ndarray, y_train: np.ndarray, loss_func: str, *args, 
    x_valid: np.ndarray=None, y_valid: np.ndarray=None, 
    loss_func_grad: Callable[[float, float], float]=None, 
    loss_func_eval: Callable[[float, float], float]=None, 
    **kwargs
):
    logger.info("START")
    obj = _train(
        params, x_train, y_train, *args, 
        x_valid=x_valid, y_valid=y_valid,
        loss_func=loss_func, loss_func_grad=loss_func_grad,
        loss_func_eval=loss_func_eval,
        func_train=lgbtune.train,
        **kwargs
    )
    logger.info("END")
    return obj


def tune(
    x_train: np.ndarray, y_train: np.ndarray, 
    loss_func, n_trials: int, params_add: dict=None,
    x_valid: np.ndarray=None, y_valid: np.ndarray=None, 
    loss_func_grad: Callable[[float, float], float]=None, 
    loss_func_eval: Callable[[float, float], float]=None, 
    **kwargs
):
    logger.info("START")
    import optuna
    from kkutils.util.ml import create_optuna_params
    params = {
        "task"             : ["const", "train"],
        'verbosity'        : ["const", -1],
        'boosting'         : ["const", "gbdt"],
        "n_jobs"           : ["const", 1],
        "random_seed"      : ["const", 1],
        "learning_rate"    : ["const", 0.03],
        "max_depth"        : ["const", -1],
        "num_iterations"   : ["const", 1000],
        'bagging_freq'     : ["const", 0],
        'num_leaves'       : ["const", 100],
        'lambda_l1'        : ["log", 1e-8, 100.0],
        'lambda_l2'        : ["log", 1e-8, 100.0],
        "min_hessian"      : ["log", 1e-5, 100.0],
        'feature_fraction' : ["float", 0.01, 0.8],
        'bagging_fraction' : ["float", 0.01, 0.8],
        'min_child_samples': ["int", 1, 100],
    }
    if params_add is not None:
        for x, y in params_add.items():
            params[x] = y
    if len(y_train.shape) == 2:
        params["num_class"] = ["const", y_train.shape[-1]]
    def objective(
        trial, params=None, x_train=None, y_train=None,
        x_valid=None, y_valid=None,
        loss_func=None, loss_func_grad=None,
        loss_func_eval=None, **kwargs
    ):
        _params = create_optuna_params(params, trial)
        gbm = _train(
            _params, x_train=x_train, y_train=y_train,
            x_valid=x_valid, y_valid=y_valid,
            loss_func=loss_func, loss_func_grad=loss_func_grad,
            loss_func_eval=loss_func_eval,
            func_train=lgb.train,
            **kwargs
        )
        func = func_embed(loss_func_eval, calc_type="mean")
        val  = func(gbm.predict(x_valid), y_valid)
        return val
    _objective = partial(
        objective, 
        params=params,
        x_train=x_train, y_train=y_train,
        x_valid=x_valid, y_valid=y_valid,
        loss_func=loss_func, loss_func_grad=loss_func_grad,
        loss_func_eval=loss_func_eval,
        **kwargs
    )
    study = optuna.create_study(
        study_name='optuna_'+datetime.datetime.now().strftime("%Y%m%d%H%M%S"),
        storage='sqlite:///optuna_'+datetime.datetime.now().strftime("%Y%m%d%H%M%S")+'.db',
    )
    # パラメータ探索
    study.optimize(_objective, n_trials=n_trials)
    # 結果を保存する
    df_optuna = pd.DataFrame()
    for i_trial in study.trials:
        sewk = pd.Series(i_trial.params)
        sewk["value"]  = i_trial.value
        df_optuna = df_optuna.append(sewk, ignore_index=True)
    dict_param_ret = {}
    for key, val in  params.items():
        if val[0] == "const": dict_param_ret[key] = val[-1]
    for key, val in  study.best_params.items():
        dict_param_ret[key] = val
    logger.info("END")
    return df_optuna, dict_param_ret

from sklearn.exceptions import NotFittedError
from typing import List
from lightgbm import LGBMClassifier, LGBMRegressor, LGBMModel
from lightgbm.callback import record_evaluation, EarlyStopException, _format_eval_result

# local package
from kkutils.util.numpy import softmax, sigmoid
from kkutils.util.com import set_logger, MyLogger
logger = set_logger(__name__)


__all__ = [
    "KkLGBMModelBase",
    "KkLGBMClassifier",
    "KkLGBMRegressor",
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
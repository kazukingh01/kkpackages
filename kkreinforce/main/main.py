
# local package
from kkreinforce.lib.qlearn import TSPModel
from kkimagemods.util.logger import set_logger, set_loglevel

if __name__ == "__main__":
    #set_loglevel(log_level="debug")
    model = TSPModel(0.2, 0.1, 0.5)
    model.play(output="result_0.html")
    model.train(n_episode=100)
    model.play(output="result_100.html")
    model.train(n_episode=1000)
    model.play(output="result_1000.html")
    model.train(n_episode=5000)
    model.play(output="result_5000.html")

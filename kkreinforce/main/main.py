
# local package
from kkreinforce.lib.tsp2 import TSPModel, TSPModel2, TSPModel3, TSPModel4
from kkimagemods.util.logger import set_logger, set_loglevel

if __name__ == "__main__":
    #set_loglevel(log_level="debug")
    """
    # 行った国の履歴を考慮しないTSP
    model = TSPModel(0.5, 0.5, 0.5)
    model.play(output="result_0.html")
    model.train(n_episode=100)
    model.play(output="result_100.html")
    model.train(n_episode=1000)
    model.play(output="result_1000.html")
    model.train(n_episode=5000)
    model.play(output="result_5000.html")
    model.train(n_episode=10000)
    model.play(output="result_10000.html")

    # 行った国の履歴を考慮するTSP
    model = TSPModel2(0.5, 0.5, 0.5, n_capital=10)
    model.play(output="result_0.html")
    model.train(n_episode=100)
    model.play(output="result_100.html")
    model.train(n_episode=1000)
    model.play(output="result_1000.html")
    model.train(n_episode=5000)
    model.play(output="result_5000.html")
    model.train(n_episode=10000)
    model.play(output="result_10000.html")

    # 行った国の履歴を考慮しないDQNのTSP
    model = TSPModel3(0.5, 0.5, 0.25)
    model.play(output="result_0.html")
    model.train(n_episode=100)
    model.play(output="result_100.html")
    model.train(n_episode=1000)
    model.play(output="result_1000.html")
    """
    # 行った国の履歴を考慮するDQNのTSP
    model = TSPModel4(0.10, 0.5, 0.9, n_capital=10)
    model.play(output="result_0.html")
    model.train(n_episode=100)
    model.play(output="result_100.html")
    for _ in range(100):
        model.train(n_episode=100)
        model.play(output="result_1000.html")
    """
    """

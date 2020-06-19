
# local package
from kkreinforce.lib.tsp import TSPModel, TSPModel2, TSPModel3, TSPModel4, TSPModel5, TSPModel6
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
    for _ in range(1000):
        model.train(n_episode=20)
        model.play(output="result_1000.html")

    model = TSPModel4(0.1, 0.5, 0.95, n_capital=10)
    for _ in range(1000):
        model.train(n_episode=20)
        model.play(output="result_1000.html")

    model = TSPModel5(0.25, 0.5, 0.95, n_capital=10)
    for _ in range(1000):
        model.train(n_episode=20)
        model.play(output="result_1000.html")
    """

    model = TSPModel6(0.05, 0.5, 0.98, n_capital=20)
    for _ in range(1000):
        model.train(n_episode=20)
        model.play(output="result_1000.html")
    """
    # 行った国の履歴を考慮するDQNのTSP
    model = TSPModel4(0.05, 0.5, 0.98, n_capital=10)
    model.play(output="result_0.html")
    model.train(n_episode=100)
    model.play(output="result_100.html")
    for _ in range(1000):
        model.train(n_episode=20)
        model.play(output="result_1000.html")

    model = TSPModel5(0.05, 0.5, 0.98, n_capital=20)
    model.play(output="result_0.html")
    model.train(n_episode=100)
    model.play(output="result_100.html")
    for _ in range(1000):
        model.train(n_episode=20)
        model.play(output="result_1000.html")
    """

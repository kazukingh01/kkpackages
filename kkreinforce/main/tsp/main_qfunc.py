from kkreinforce.models.tsp.tsp_qtable import TSPModel, TSPModel2
from kkreinforce.models.tsp.tsp_dqn import TSPModel3, TSPModel4, TSPModel5, TSPModel6, TSPModel7


if __name__ == "__main__":
    model = TSPModel(0.5, 0.5, 0.5, file_csv="../../data/s59h30megacities_utf8.csv", n_capital=None)
    for _ in range(100):
        model.train(n_episode=20)
        model.play(output="result_100.html")

    model = TSPModel2(0.5, 0.5, 0.25, file_csv="../../data/s59h30megacities_utf8.csv", n_capital=10)
    for _ in range(1000):
        model.train(n_episode=20)
        model.play(output="result_100.html")

    model = TSPModel3(0.5, 0.5, file_csv="../../data/s59h30megacities_utf8.csv", n_capital=None)
    for _ in range(100):
        model.train(n_episode=20)
        model.play(output="result_100.html")

    model = TSPModel4(0.5, 0.5, file_csv="../../data/s59h30megacities_utf8.csv", n_capital=None)
    for _ in range(100):
        model.train(n_episode=20)
        model.play(output="result_100.html")

    model = TSPModel5(0.1, 0.5, file_csv="../../data/s59h30megacities_utf8.csv", n_capital=50)
    for _ in range(100):
        model.train(n_episode=20)
        model.play(output="result_100.html")

    model = TSPModel6(0.1, 0.99, file_csv="../../data/s59h30megacities_utf8.csv", n_capital=10)
    for _ in range(1000):
        model.train(n_episode=20)
        model.play(output="result_100.html")

    model = TSPModel7(0.1, 0.99, file_csv="../../data/s59h30megacities_utf8.csv", n_capital=10)
    for _ in range(1000):
        model.train(n_episode=20)
        model.play(output="result_100.html")

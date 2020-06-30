from kkreinforce.lib.tsp2 import TSPModel


if __name__ == "__main__":
    model = TSPModel(0.2, 0.5, 0.98, n_capital=10)
    for _ in range(1000):
        model.train(n_episode=20)
        model.play(output="result_1000.html")

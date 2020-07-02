from kkreinforce.models.tsp.tsp_pg import TSPModel
from kkimagemods.util.logger import set_logger, set_loglevel

if __name__ == "__main__":
    set_loglevel(log_level="debug")
    model = TSPModel(1.0, file_csv="../../data/s59h30megacities_utf8.csv", n_capital=10)
    for _ in range(1000):
        model.train(n_episode=20)
        model.play(output="result_100.html")

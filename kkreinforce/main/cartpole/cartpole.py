from kkreinforce.models.cartpole.cartpole import CartPole_QTable, CartPole_DNN
from kkimagemods.util.logger import set_loglevel


if __name__ == "__main__":
    #set_loglevel(log_level="debug")

    """
    model = CartPole_QTable(alpha=0.5, gamma=0.99, epsilon=0.99)
    for _ in range(100):
        model.play(save_gif_path=f"./cartpole_qtable_{model.episode}.gif")
        model.train(n_episode=100)
    model.env.close()
    """

    model = CartPole_DNN(gamma=0.99, epsilon=0.99)
    for _ in range(100):
        model.play(save_gif_path=f"./cartpole_dqn_{model.episode}.gif")
        model.train(n_episode=20)
    model.env.close()

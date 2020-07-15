import cv2, time
from kkreinforce.models.cartpole.cartpole import CartPole_QTable, CartPole_DNN, CartPole_CNN, CartPole_CNN2, CartPole_CNN3, CartPole_PG
from kkimagemods.util.logger import set_loglevel


if __name__ == "__main__":
    set_loglevel(name="kkreinforce.lib.kkrl", log_level="debug")
    set_loglevel(name="kkreinforce.lib.kknn", log_level="debug")
    set_loglevel(name="kkreinforce.lib.qlearn", log_level="debug")
    #set_loglevel(name="kkreinforce.models.cartpole.cartpole", log_level="debug")

    """
    model = CartPole_QTable(alpha=0.5, gamma=0.99, epsilon=0.99)
    for _ in range(100):
        model.play(save_gif_path=f"./cartpole_qtable_{model.episode}.gif")
        model.train(n_episode=100)
    model.env.close()
    """

    """
    model = CartPole_DNN(gamma=0.99, epsilon=0.99)
    for _ in range(100):
        model.play(save_gif_path=f"./cartpole_dqn_{model.episode}.gif")
        model.train(n_episode=20)
    model.env.close()

    model = CartPole_CNN3(gamma=0.99, epsilon=0.5)
    for _ in range(1000):
        model.play(save_gif_path=f"./cartpole_cnn_{model.episode}.gif", display=True)
        model.train(n_episode=10)
    model.env.close()
    """

    model = CartPole_PG(gamma=0.9)
    for _ in range(1000):
        model.play(save_gif_path=f"./cartpole_pg_{model.episode}.gif", display=True)
        model.train(n_episode=10)
    model.env.close()

    """
    for i in range(10):
        cv2.imshow("test", model.state_now[i])
        cv2.waitKey(0)
    """

from fqtrl.environment import BundEnvironment
from fqtrl.utils import create_mock_data, plot_price_and_position, plot_position
import fqtrl.montecarlo as mc
import pprint
import matplotlib.pyplot as plt
import numpy as np

market_features = ["indicator"]
mock_data = create_mock_data(60, 5)
#mock_data.bid = np.array([0, 1, 2, 3, 4]) / 100
#mock_data.ask = np.array([0, 1, 2, 3, 4]) / 100
plot_price_and_position(mock_data)
mock_symbol = "BD#"
mock_datestr = "2019-02-04"

n_episodes = 1000
alpha = 0.1
gamma = 0.9
epsilon = 1.0
epsilon_decay_rate = 0.99

env = BundEnvironment(mock_symbol, mock_datestr, mock_data, market_features)
# episode = mc.generate_stochastic_episode(env)
# Q = mc.mc_prediction_q(env, n_episodes, mc.generate_stochastic_episode)
# episode_from_Q = mc.generate_episode_from_Q(env, Q)
# mc.update_Q(env, episode_from_Q, Q, alpha, gamma)
sum_R_over_episodes, Q, Pi_star, A_from_last_episode = mc.control(
    env, n_episodes, alpha, gamma, epsilon, epsilon_decay_rate
)


plt.title("Reward over number of episodes")
plt.xlabel("Episode number")
plt.ylabel("Reward")
plt.plot(sum_R_over_episodes, marker=".", linestyle="")
filename = "sum_R_over_episodes.png"
plt.savefig(filename)
plt.close()
plot_position(mock_data, A_from_last_episode)

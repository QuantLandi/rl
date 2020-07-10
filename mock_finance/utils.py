import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


np.random.seed(42)


def create_mock_data(sample_size, n_switchpoints):
    switchpoint_step = sample_size / n_switchpoints
    switchpoints = np.arange(0, sample_size, switchpoint_step)
    price_at_switchpoints = (
        np.random.randint(17434, 17474, len(switchpoints)) / 100
    )
    bid_price = np.interp(
        np.arange(sample_size), switchpoints, price_at_switchpoints
    )
    optimal_position = np.sign(
        np.diff(np.concatenate([np.array([0]), bid_price]))
    )
    ask_price = bid_price + 0.01

    optimal_action = np.diff(np.concatenate([np.array([0]), optimal_position]))
    reverse_position = np.where(
        np.abs(optimal_action) > 1, np.sign(optimal_action), 0
    )
    optimal_action = np.sign(optimal_action) + np.roll(reverse_position, 1)
    optimal_action = np.where(optimal_action == -1.0, 2, optimal_action)

    data = np.array(
        [optimal_position, optimal_action, bid_price, ask_price]
    ).transpose()
    columns = ["indicator", "optimal_action", "bid", "ask"]
    start = pd.to_datetime("2019-02-04 09:00")
    end = start + timedelta(seconds=sample_size-1)
    index = pd.date_range(
        start=start,
        end=end,
        freq="s",
        tz="Europe/Paris",
    )

    mock_data = pd.DataFrame(data, index, columns).dropna()
    mock_data[["bid", "ask"]] = round(mock_data[["bid", "ask"]], 2)
    return mock_data


def plot_price_and_position(mock_data):
    mid_price = (mock_data.bid.values + mock_data.ask.values) / 2
    optimal_position = mock_data.indicator.values
    t = np.arange(len(mid_price))
    data1 = mid_price
    data2 = optimal_position

    fig, ax1 = plt.subplots(figsize=(15, 5))

    color = "tab:red"
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("Price", color=color)
    ax1.plot(t, data1, color=color)
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = "tab:blue"
    ax2.set_ylabel(
        "Optimal Position", color=color
    )  # we already handled the x-label with ax1
    ax2.plot(t, data2, "o", markersize=1, color=color)
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title("Price and Optimal Position")
    data_plot_filename = "mock_data.png"
    plt.savefig(data_plot_filename)
    plt.close()


def plot_episode_rewards(
    episode_rewards,
    #     last_100_scores_rolling_means,
    #     episode_count,
    buffer_size,
    batch_size,
    gamma,
    tau,
    lr,
    update_every,
    qnetwork_local
):
    fig, ax = plt.subplots(figsize=(20, 10))
    textstr = (
        "max(last_100_scores_means): {}\nepisode_count: {}\n"
        + "buffer_size: {}\nbatch_size: {}\ngamma: {}\n"
        + "tau: {}\nlr: {}\nupdate_every: {}\nqnetwork_local: {} "
    )
    #     last_100_rewards = None
    textstr = textstr.format(
        round(np.max(episode_rewards), 2),
        #         step_count,

        buffer_size,
        batch_size,
        gamma,
        tau,
        lr,
        update_every, 
        qnetwork_local
    )
    ax.plot(
        np.arange(len(episode_rewards)),
        episode_rewards,
        "o",
        label="Single episode reward",
        markersize=3,
    )
    #     idxs = np.arange(len(last_100_scores_rolling_means)
    #     last_100_scores_rolling_means = np.where(
    #         idxs < 100,
    #         np.nan,
    #         last_100_scores_rolling_means
    #     )
    #     ax.plot(
    #         np.arange(
    #             idxs,
    #             last_100_scores_rolling_means,
    #             label="Last 100 scores rolling mean",
    #             linewidth=5
    #         )

    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax.text(
        0.05,
        0.95,
        textstr,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=props,
    )
    ax.legend(loc="upper right")
    ax.set_title("Score over number of episodes")
    ax.set_xlabel("Episode number")
    ax.set_ylabel("Score")
    filename = "scores_over_episodes.png"
    plt.savefig(filename)
    plt.close()


def plot_actions(mock_data, actions):
    mid_price = (mock_data.bid.values + mock_data.ask.values) / 2
    t = np.arange(len(mid_price))
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.set_xlabel("Step")
    ax.set_ylabel("Price")
    ax.set_title("Actions during episode")
    ax.plot(t, mid_price)
    last_buy_idx, last_sell_idx = None, None
    for i, action in enumerate(actions):
        if action == 1:
            ax.axvline(i, color="#2ca02c")
            last_buy_idx = i
        elif action == 2:
            ax.axvline(i, color="#d62728")
            last_sell_idx = i
    if last_buy_idx:
        ax.axvline(last_buy_idx, color="#2ca02c", label="buy")
    if last_sell_idx:
        ax.axvline(last_sell_idx, color="#d62728", label="sell")
    ax.plot(t, mid_price, color="#ff7f0e")
    ax.legend()
    filename = "actions.png"
    plt.savefig(filename)
    plt.close()


def plot_position(mock_data, actions):
    mid_price = (mock_data.bid.values + mock_data.ask.values) / 2
    t = np.arange(len(mid_price))
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.set_xlabel("Step")
    ax.set_ylabel("Price")
    ax.set_title("Position over episode")
    ax.set_xticks(t)
    ax.plot(t, mid_price)
    actions = np.array(actions)
    actions = np.where(actions == 2, -1, actions)
    positions = np.cumsum(actions)[:-1]
    ax.pcolorfast((0, len(mock_data)-1), ax.get_ylim(), positions[np.newaxis],
                  cmap='RdYlGn', vmin=-1, vmax=1, alpha=0.3)
    filename = "position_over_episode.png"
    plt.savefig(filename)
    plt.close()

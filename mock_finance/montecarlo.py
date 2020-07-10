import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .market import Action, Position, IDLE, BUY, SELL, FLAT, LONG, SHORT
from tqdm import tqdm


np.random.seed(42)


def get_policy(env, Q, s, A, epsilon):
    """ obtains action probabilities corresponding to epsilon-greedy policy """
    n_actions = len(Action)
    is_valid_action = env.get_valid_action_mask()
    n_valid_actions = sum(is_valid_action)
    t, position = s[0], s[1]
    pi_s = np.zeros(n_actions)
    Q_s = np.where(is_valid_action, Q[t, position, :], np.NINF)
    argmax_Q_s = np.argmax(Q_s)
    max_Q = Q_s[argmax_Q_s]
    # deal with edge case of multiple max to avoid arbitrary argmax selection
    is_multiple_max = sum(max_Q == Q_s) > 1
    if is_multiple_max:
        max_Q_idxs = [i for i, Q_sa in enumerate(Q_s) if Q_sa == max_Q]
        greedy_a = np.random.choice(max_Q_idxs)
        pi_s[greedy_a] = 1
    else:
        greedy_a = argmax_Q_s
        pi_s[greedy_a] = 1 - epsilon
        pi_s += np.where(is_valid_action, epsilon / n_valid_actions, 0.0)
    return pi_s


def generate_episode_from_Q(env, Q, N, epsilon=0.0):
    """ generates an episode from following the epsilon-greedy policy """
    episode = []
    env.reset()
    s = env.get_state_vector()
    A = np.array([0, 1, 2])
    sum_R = 0
    n_steps = len(env.df)
    for _ in np.arange(n_steps):
        t, position = s[0], s[1]
        pi_s = get_policy(env, Q, s, A, epsilon)
        a = Action(np.random.choice(A, p=pi_s))
        # track how many times the (state, action) was visited
        N[t, position, a] += 1
        # collect step reward and add it to cumulative reward
        r = env.do_step(a)
        sum_R += r
        # collect (state, action, reward) tuple
        episode.append((s, a, r))
        s_prime = env.get_state_vector()
        s = s_prime
    return episode, sum_R


def update_Q(env, episode, Q, alpha, gamma):
    """ updates action-value function estimate using most recent episode """
    S, A, R = zip(*episode)
    # prepare for discounting
    gamma_power_t = np.array([gamma ** t for t in range(len(R))])
    for s in S:
        t, position = s[0], s[1]
        a = A[t]
        #s, a = str(S[t]), A[t]
        # compute episode reward
        G_t = sum(R[t:] * gamma_power_t[:len(R)-t])
        # update Q-table using Bellman equation
        with open("mc.log", "a") as q_log:
            old_Q_sa = np.round(Q[t, position, a], 2)
            Q_sa = Q[t, position, a]
            Q[t, position, a] = Q_sa + alpha * (G_t - Q_sa)
            delta_Q_sa = np.round(alpha * (G_t - Q_sa), 2)
            #Q[s][a] = Q[s][a] + alpha * (G_t - Q[s][a])
            new_Q_sa = np.round(Q[t, position, a], 2)
            q_log.write(
                "t: "+str(t)+" "+str(Position(position))+" "+str(a)+"\n"+
                "old_Q_sa: \n"+str(old_Q_sa)+"\n"+
                "Q[t, position, a] = Q_sa + alpha * (G_t - Q_sa)\n"+
                "G_t: "+str(np.round(G_t, 2))+", "+"old_Q_sa: "+str(old_Q_sa)+", "+
                "alpha: "+str(alpha)+", "+
                "delta_Q_sa: "+str(delta_Q_sa)+"\n"+
                "delta_Q: \n"+str(new_Q_sa-old_Q_sa)+"\n"+
                "new_Q_sa: \n"+str(new_Q_sa)+"\n\n"
                )
    return Q


def control(
    env, n_episodes, alpha, gamma=1.0, epsilon=0.0, eps_decay_rate=0.99,
):
    n_steps, n_positions, n_actions = len(env.df), len(Position), len(Action)
    # initialize empty dictionary of arrays
    Q = np.zeros((n_steps, n_positions, n_actions))
    N = np.zeros((n_steps, n_positions, n_actions))
    sum_R_over_episodes = np.zeros(n_episodes)
    with open("mc.log", "w") as q_log:
        q_log.write("")
    # loop over episodes
    for i in tqdm(range(n_episodes)):
        with open("mc.log", "a") as q_log:
            q_log.write("Episode "+str(i)+"\n\n") 
        # generate episode by following epsilon-greedy policy
        episode, sum_R = generate_episode_from_Q(env, Q, N, epsilon)
        epsilon *= eps_decay_rate
        # update action-value function estimate using episode
        Q = update_Q(env, episode, Q, alpha, gamma)
        # set Q-table entries of illegal actions as np.NINF
        Q[:, [1,2], [1,2]] = np.NINF
        # compute policy for all states
        Q_argmax = np.argmax(Q, axis=2)
        Pi_star = (
            np.arange(len(Action)) == Q_argmax[...,None]
        ).astype(int)
        # collect cumulative reward of episode for visualization purposes
        sum_R_over_episodes[i] = sum_R
        with open("mc.log", "a") as q_log:
            q_log.write(
                "Pi_star:\n"+str(Pi_star)+
                "\n\nQ:\n"+str(np.round(Q, 2))+
                "\n\nN:\n"+str(N)+"\n\n"
            )
    # collect actions from last episode for plotting purposes
    A_from_last_episode = [int(experience[1]) for experience in episode]
    return sum_R_over_episodes, Q, Pi_star, A_from_last_episode

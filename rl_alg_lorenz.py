import os.path
import random
import time
import numpy as np
from s_discretization import random_initial_conditions, states_clustering, get_energy_clust, show_ek, get_laminar_states
from a_discretization import get_action_space, get_action_limits, get_B
from mfe_model import MoehlisFaisstEckhardtModelControl, rk4_timestepping_control
from rl_alg import Environment, q_learning
from lorenz_model import LorenzModelControl
from s_discr_lorenz import show_lorenz_discr_2D, show_lorenz_discr, show_lorenz_2D, show_lorenz

from numpy.linalg import norm

class LorenzEnvironment(Environment):
    def __init__(self, action_space, m, clust_model, n_steps):
        self.action_space = action_space
        self.state_space_model = clust_model
        self.n_a = len(action_space)
        self.n_s = len(clust_model.cluster_centers)
        self.n_steps = n_steps

        self.model = m
        self.cur_state = np.zeros(self.model.dim)

        self.delta_t = 0.001
        self.eps = 1e-5
        self.time = 0
        self.T = 100

        self.real_trajectory = np.zeros((self.T+1, self.model.dim))

        self.stable_state = [0, 0, 0]

    def set_initial_conditions(self, is_random=True, seed=None, limit=1):
        if is_random:
            return random_initial_conditions(self.model.dim, seed, limit=limit, is_mfe=False)
        start_a = np.random.choice(self.start_states)
        # start_a = self.start_states[0]
        return self.state_space_model.cluster_centers[start_a]

    def get_reward(self, action):
        return -norm(self.cur_state - self.stable_state)

    def show_trajectory(self, real=False, discr=None):
        if real:
            show_lorenz_2D(self.real_trajectory)
            show_lorenz(self.real_trajectory)
        if discr is not None:
            show_lorenz_discr_2D(self.model, self.state_space_model, discr)
            show_lorenz_discr(self.model, self.state_space_model, discr)



def count_delta(q_n, q_o):
    av_d = 0.0
    for i in range(len(q_n)):
        for j in range(len(q_n[0])):
            av_d += abs(q_n[i][j] - q_o[i][j])
    return av_d/(len(q_n)*len(q_n[0]))


def init_q_table(n_s, n_a, init_num):
    q = np.full([n_s, n_a], init_num)
    st_state = env.state_space_model.transform(np.array([env.stable_state]))
    q[st_state] = np.zeros(n_a)
    return q


if __name__ == "__main__":
    model = LorenzModelControl(10, 28, 8 / 3)

    trajectory = np.loadtxt('time_series/trajectory_for_clustering_lorenz.txt')

    n_states = 300
    clust_u, assign_u = states_clustering('kmeans_uniform', trajectory, n_iter_max=1000, n_cl=n_states)

    a = 10
    a_comp = 2
    perc_range = 80
    a_vec, action_space = get_action_space(get_action_limits(get_B(model, trajectory),perc_range), a, num_of_a=a_comp)

    # print(len(action_space))
    alpha = 0.1
    gamma = 0.95
    epsilon = 0.15

    n_steps_rk = 1000

    env = LorenzEnvironment(action_space, model, clust_u, n_steps_rk)

    n_episodes = 10
    save_res = True
    filename = f'q_tables/lorenz_q_table_a_{a}_acomp_{a_comp}_rk_{n_steps_rk}_s_{n_states}_perc{perc_range}.gz'
    time_filename = f"simulation_time/lorenz_time_{n_states}s_{a_comp}comp_{a}a_perc{perc_range}"

    if not os.path.exists(filename):
        np.savetxt(filename, init_q_table(env.n_s, env.n_a, -1))

    seed = 6
    q_learning(env, filename, time_filename, save_res, n_episodes, epsilon, alpha, gamma,
               is_rand=True, seed=seed, show_real=True, show_discr=False, a_comp=a_comp)
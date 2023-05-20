import os.path
import random
import time
import numpy as np
from s_discretization import random_initial_conditions, states_clustering, get_energy_clust, show_ek, get_laminar_states
from a_discretization import get_action_space, get_action_limits, get_B
from mfe_model import MoehlisFaisstEckhardtModelControl, rk4_timestepping_control
from rl_alg import Environment, q_learning
from lorenz_model import LorenzModelControl

from numpy.linalg import norm


def special_ic(cur_clust, cur_assign, model):
    ke = get_energy_clust(cur_clust, cur_assign, model)
    start_states = []
    start_states_a = []
    for t in range(len(ke)):
        if 19 < ke[t] < 20:
            if cur_assign[t] not in start_states:
                start_states.append(cur_assign[t])
    return start_states

class LorenzEnvironment(Environment):
    def __init__(self, action_space, m, clust_model):
        self.action_space = action_space
        self.state_space_model = clust_model
        self.n_a = len(action_space)
        self.n_s = len(clust_model.cluster_centers)

        self.model = m
        self.cur_state = np.zeros(self.model.dim)

        # self.start_states = special_ic(clust_u, assign_u, model)

        self.delta_t = 0.001
        self.eps = 1e-5
        self.time = 0
        self.T = 1000

        self.stable_states = np.array([[0,0,0], [np.sqrt(self.model.beta * (self.model.Ra - 1)), np.sqrt(self.model.beta * (self.model.Ra - 1)), self.model.Ra - 1],
                              [-np.sqrt(self.model.beta * (self.model.Ra - 1)), -np.sqrt(self.model.beta * (self.model.Ra - 1)), self.model.Ra - 1]])

    def set_initial_conditions(self, is_random=True, seed=None):
        if is_random:
            return random_initial_conditions(self.model.dim, seed, is_mfe=False)
        start_a = np.random.choice(self.start_states)
        # start_a = self.start_states[0]
        return self.state_space_model.cluster_centers[start_a]

    # def reset(self, is_rand=True, seed=None):
    #     self.time = 0
    #     self.cur_state = self.set_initial_conditions(is_rand, seed)
    #     # print('ic:', self.cur_state)
    #     return self.state_space_model.transform(np.array([self.cur_state]))

    def get_reward(self, action):
        # print(state)
        # print("dist: ", norm(self.cur_state - self.model.laminar_state))
        # print("a: ", norm(action))
        # print(-(norm(self.cur_state - self.model.laminar_state) + 0.5*norm(action)))
        r = np.zeros(len(self.stable_states))
        for a in range(len(self.stable_states)):
            r[a] = -10*norm(self.cur_state - self.stable_states[a])
        return np.min(r) #- 5*norm(action) #- 0.005*self.time # штраф за т?
    #
    # def lam_stop_condition(self, reward):
    #     return -self.eps <= reward <= self.eps
    #
    # def time_stop_condition(self):
    #     return self.time >= self.T

    # def step(self, action_num, n_steps=1000):
    #     self.time += n_steps*self.delta_t
    #     if (self.time % 100) == 0:
    #         print("t = ", self.time)
    #     self.cur_state = rk4_timestepping_control(self.model, self.cur_state, self.action_space[action_num],
    #                                               delta_t=self.delta_t, n_steps=n_steps)[-2]
    #     next_state = self.state_space_model.transform(np.array([self.cur_state]))
    #     reward = self.get_reward(self.action_space[action_num])
    #
    #     if self.time_stop_condition():
    #         return next_state, reward, True
    #     elif self.lam_stop_condition(reward):
    #         print('Laminar state')
    #         return next_state, reward+10, True
    #     return next_state, reward, False

def count_delta(q_n, q_o):
    av_d = 0.0
    for i in range(len(q_n)):
        for j in range(len(q_n[0])):
            av_d += abs(q_n[i][j] - q_o[i][j])
    return av_d/(len(q_n)*len(q_n[0]))
    # max_d = 0.0
    # for i in range(len(q_n)):
    #     for j in range(len(q_n[0])):
    #         d = abs(q_n[i][j] - q_o[i][j])
    #         if max_d < d:
    #             max_d = d
    # return max_d

#
# def q_learning(env, filename, save_res, episodes=1, epsilon=0.15, alpha=0.1, gamma=0.9, n_steps_rk=1000, is_rand=True, seed=None):
#     q_table = np.loadtxt(filename)
#
#     start_time = time.time()
#     for i in range(episodes):
#         print("Iteration ", i)
#         q_old = np.copy(q_table)
#         state = env.reset(is_rand, seed)
#         reward = 0
#         done = False
#         states_traj = np.array([state])
#
#         while not done:
#             if random.uniform(0, 1) < epsilon:
#                 action = random.randrange(len(env.action_space))
#             else:
#                 action = np.argmax(q_table[state])
#
#             next_state, reward, done = env.step(action, n_steps_rk)
#
#             new_value = (1 - alpha) * q_table[state, action] + alpha * (reward + gamma * np.max(q_table[next_state]))
#             q_table[state, action] = new_value
#
#             state = next_state
#             states_traj = np.append(states_traj, state)
#         if save_res:
#             # np.savetxt(f'time_series/ep_trajectories_{i}.txt', states_traj, fmt='%1u')
#             np.savetxt(filename, q_table)
#
#         print(count_delta(q_table, q_old))
#         if env.time < env.T:
#             show_ek(None, [model, clust_u, states_traj, None, f'{i}', n_steps_rk*env.delta_t])
#         show_ek(None, [model, clust_u, states_traj, None, f'{i}', n_steps_rk*env.delta_t])
#
#     print("Training finished.\n")
#     tt = time.time() - start_time
#     print(f"{tt // 60} min, {tt % 60} sec")
#
#     if save_res:
#         with open(f"simulation_time/time_{n_states}s_{a_comp}comp_{a}a_perc{perc_range}_continuous", "a") as time_file:
#             t_time = str(round(tt/60, 2))
#             time_file.write(f"{t_time} {env.T} {episodes}\n")

def init_q_table(n_s, n_a, init_num):
    q = np.full([n_s, n_a], init_num)
    for i in range(len(env.stable_states)):
        st_state = env.state_space_model.transform(np.array([env.stable_states[i]]))
        q[st_state] = np.zeros(n_a)
    return q


if __name__ == "__main__":
    model = LorenzModelControl(10, 28, 8 / 3)

    trajectory = np.loadtxt('time_series/trajectory_for_clustering_lorenz.txt')

    n_states = 300
    clust_u, assign_u = states_clustering('kmeans_uniform', trajectory, n_iter_max=1000, n_cl=n_states)

    a = 5
    a_comp = 3
    perc_range = 80
    a_vec, action_space = get_action_space(get_action_limits(get_B(model, trajectory),perc_range), a, num_of_a=a_comp)

    env = LorenzEnvironment(action_space, model, clust_u)

    alpha = 0.1
    gamma = 0.95
    epsilon = 0.15

    n_steps_rk = 5000

    n_episodes = 10
    save_res = True
    filename = f'q_tables/lorenz_q_table_a_{a}_acomp_{a_comp}_rk_{n_steps_rk}_s_{n_states}_perc{perc_range}_continuous.gz'  # + диапазон а
    time_filename = f"simulation_time/lorenz_time_{n_states}s_{a_comp}comp_{a}a_perc{perc_range}_continuous"

    if not os.path.exists(filename):
        np.savetxt(filename, init_q_table(env.n_s, env.n_a, -0.5))

    seed = 6
    q_learning(env, filename, time_filename, save_res, n_episodes, epsilon, alpha, gamma, n_steps_rk, True, seed)

import os.path
import random
import time
import numpy as np
from s_discretization_MFE import random_initial_conditions, states_clustering, get_energy_clust, show_ek, get_laminar_states
from a_discretization import get_action_space, get_action_limits, get_B, show_B
from mfe_model import MoehlisFaisstEckhardtModelControl, rk4_timestepping_control
from rl_alg import Environment, q_learning
from lorenz_model import LorenzModelControl
from s_discr_lorenz import show_lorenz_discr_2D, show_lorenz_discr, show_lorenz_2D, show_lorenz

import matplotlib.pyplot as plt
from numpy.linalg import norm
from copy import copy

class LorenzEnvironment(Environment):
    def __init__(self, action_space, m, clust_model, n_steps, traj):
        self.action_space = action_space
        self.state_space_model = clust_model
        self.n_a = len(action_space)
        self.n_s = len(clust_model.cluster_centers)
        self.n_steps = n_steps

        self.model = m
        self.cur_state = np.zeros(self.model.dim)

        self.delta_t = 0.001
        self.eps = 5e-13
        self.epsd = 1e-4
        self.time = 0
        self.T = 500

        self.t_left = 0
        self.lam_t = 0
        self.real_trajectory = np.zeros((int((self.T)/(self.delta_t*self.n_steps))+1, self.model.dim))
        self.da = np.zeros((int((self.T)/(self.delta_t*self.n_steps))+1, self.model.dim))
        # self.u = np.zeros((self.T + 1, self.model.dim))

        def get_p(da):
            da_T = da.T
            p_max = np.zeros(len(da_T))
            for i in range(len(da_T)):
                p_max[i] = max(abs(np.min(da_T[i])), np.max(da_T[i]))
            return p_max


        self.stable_state = [-np.sqrt(8/3 * 27), -np.sqrt(8/3 * 27), 27]
        self.stable_states = [[0,0,0],[np.sqrt(8/3 * 27), np.sqrt(8/3 * 27), 27], [-np.sqrt(8/3 * 27), -np.sqrt(8/3 * 27), 27]]
        self.max_p = get_p(get_B(self.model, traj))
        self.lam=False


    def set_initial_conditions(self, is_random=True, seed=None, limit=1):
        if is_random:
            return random_initial_conditions(self.model.dim, seed, limit=limit, is_mfe=False)
        start_a = np.random.choice(self.start_states)
        # start_a = self.start_states[0]
        return self.state_space_model.cluster_centers[start_a]

    def reset(self, is_rand=True, seed=None):
        self.t_left = 0
        self.time = 0
        self.lam=False

        self.real_trajectory = np.zeros((int((self.T)/(self.delta_t*self.n_steps))+1, self.model.dim))
        self.da = np.zeros((int((self.T)/(self.delta_t*self.n_steps))+1, self.model.dim))
        # self.u = np.zeros((self.T + 1, self.model.dim))

        self.cur_state = self.set_initial_conditions(is_rand, seed)

        self.real_trajectory[0] = self.cur_state
        self.da[0] = model.f(self.cur_state, np.zeros(self.model.dim))
        # print('ic:', self.cur_state)
        return self.state_space_model.transform(np.array([self.cur_state]))

    def get_reward(self, action, prev_state):
        delta = abs(self.cur_state - prev_state)/self.max_p
        # dist = np.min([norm(self.stable_states[i]-self.cur_state) for i in range(len(self.stable_states))])
        dist = norm(self.stable_state-self.cur_state)
        # print(norm(delta))
        # print(norm(dist))
        # print(norm(self.cur_state - self.stable_state))
        # return -abs(norm(self.da[-1]))
        # print(-dist-0.1*norm(action))
        return -dist-0.5*norm(action) #-10*norm(delta)


    def step(self, action_num):
        self.time += (self.delta_t*self.n_steps)
        if (self.time % 100) == 0:
            print("t = ", self.time)

        state = self.state_space_model.cluster_centers[self.state_space_model.transform(np.array([self.cur_state]))]
        prev_state = self.cur_state
         # print(state)
        self.cur_state = rk4_timestepping_control(self.model, self.cur_state,
                                                  self.action_space[action_num],
                                                  delta_t=self.delta_t, n_steps=self.n_steps)[-2]
        if self.cur_state[0] <= 0:
            self.t_left += (self.delta_t*self.n_steps)
            r = 5
        else:
            self.t_left = 0
            r = -5
        next_state = self.state_space_model.transform(np.array([self.cur_state]))

        self.da[int(self.time/(self.delta_t*self.n_steps))] = model.f(self.cur_state, self.action_space[action_num])
        # self.u[int(self.time)] = np.array([0,0,self.action_space[action_num][1]*(self.model.Ra -1- self.cur_state[2])])
        reward = self.get_reward(self.action_space[action_num], prev_state)
        # print(reward)

        self.real_trajectory[int(self.time/(self.delta_t*self.n_steps))] = self.cur_state

        if self.time_stop_condition():
            print('dist',norm(self.stable_state-self.cur_state))
            print('delta', norm(self.cur_state - prev_state))
            return next_state, reward, True
        elif (norm(self.cur_state - prev_state) == 0) and (norm(action_space[action_num])==0):
            # if norm(self.cur_state - prev_state) < self.eps:
            print(f'Laminar state, t={env.time}')
            self.lam = True
            return next_state, reward, False
        # if norm(self.cur_state - prev_state) < self.epsd:
        if self.t_left > 100:
            if self.lam == False:
                print(f'Laminar state, t={self.time}, null')
                print('delta', norm(self.cur_state - prev_state))
                print('dist', np.min([norm(self.stable_states[i]-self.cur_state) for i in range(len(self.stable_states))]))
                self.lam = True
                self.lam_t =self.time
                print('da', self.da[-1])
            return next_state, reward, False
        return next_state, reward+r, False
    #
    # def show_u(self, actions, a_comp):
    #     fig, axs = plt.subplots(a_comp, figsize=(10, a_comp*2))
    #     for i in range(a_comp):
    #         axs[i].plot(np.arange(len(actions)), actions,
    #                     color='black', markersize=3, linewidth=1)
    #         axs[i].set(ylabel=f'$u_{i+1}$')
    #         axs[i].grid()
    #     plt.xlabel('$t$')
    #     plt.show()

    def show_trajectory(self, real=False, discr=None):
        if real:
            show_lorenz_2D(self.real_trajectory)
            show_lorenz(self.real_trajectory)
        if discr is not None:
            # print(self.real_trajectory)
            # print(discr)
            # print()
            # print(20/(self.delta_t*self.n_steps))
            show_lorenz_2D(self.real_trajectory[int(20/(self.delta_t*self.n_steps)):], isx=True, delta_t=self.delta_t*self.n_steps)
            # show_lorenz_discr_2D(self.model, self.state_space_model, discr)
            show_lorenz_discr(self.model, self.state_space_model, discr)
            if self.lam_t != 0:
                # show_lorenz(self.real_trajectory[:int((self.lam_t) / (self.delta_t * self.n_steps))])
                # show_lorenz(self.real_trajectory[int((self.lam_t)/(self.delta_t*self.n_steps)):])
                show_lorenz(self.real_trajectory[:int(40 / (self.delta_t * self.n_steps))])
                show_lorenz(self.real_trajectory[int(40 / (self.delta_t * self.n_steps)):])
            else:
                show_lorenz(self.real_trajectory[:int(40 / (self.delta_t * self.n_steps))])
                show_lorenz(self.real_trajectory[int(40 / (self.delta_t * self.n_steps)):])
                # show_lorenz(self.real_trajectory)
            # self.show_u(self.u, 3)
            # show_B(self.da, scaling=False)



    def show_action(self, action):
        plt.figure(figsize=(10, 3))
        t = np.arange(len(action)) * (self.delta_t*self.n_steps)
        plt.plot(t, [self.action_space[action[a]][0] for a in range(len(action))],
                    color='black', markersize=3, linewidth=1)
        plt.ylabel(f'$a_1(t)$')
        plt.grid()
        # plt.xlabel('$t$')
        plt.show()

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
    model = LorenzModelControl(10, 28, 8 / 3) # параметры модели

    # trajectory_cl = np.loadtxt('time_series/random_trajectory_for_clustering_lorenz.txt')  # траектория для кластеризации
    trajectory = np.loadtxt('time_series/trajectory_for_clustering_lorenz.txt') # траектория для определения пространства состояний
    trajectory_cl = trajectory

    perc_range = 30  # диапазон значений действий %
    a_range = get_action_limits(get_B(model, trajectory), perc_range)

    action_space = np.arange(a_range[0][0], a_range[0][1], 5)
    # action_space = np.arange(-5, 0, 1)
    action_space = np.concatenate([action_space, np.array([0.])])
    # print(len(a_arr))
    print(len(action_space))
    action_space = np.column_stack([action_space, np.zeros(len(action_space))])
    action_space = np.column_stack([action_space, np.zeros(len(action_space))])


    for a in range(len(action_space)):
        if abs(action_space[a][0]) < 0.00001:
            action_space[a][0] = 0.0
            a_null = a
            break

    print(action_space)

    n_states = 10  # число кластеров (дискретных состояний)
    a = len(action_space) # число действий для одного уравнения
    # a = 10
    a_comp = 3  # число уравнений, для которых применяется воздействие


    # Параметры алгоритма
    alpha = 0.1
    gamma = 1
    epsilon = 0

    n_episodes = 10  # Число эпизодов

    init_values = -60  # Начальные значения Q таблицы
    n_steps_rk = 100  # Число шагов метода Рунге-Кутты

    is_rand_ic = True  # Случайный выбор начальных состояний
    seed = None  # Для случайного выбора начальных состояний

    save_res = True
    # Файлы для сохранения результатов
    a_c=3
    filename = f'q_tables/x_left_lorenz_q_table_a{a}_acomp{a_c}_rk{n_steps_rk}_s{n_states}_perc{perc_range}.gz'
    time_filename = f"simulation_time/x_left_lorenz_time_{n_states}s_{a_c}comp_{a}a_perc{perc_range}"

    clust_u, assign_u = states_clustering('kmeans_uniform', trajectory_cl, n_iter_max=1000, n_cl=n_states)
    # a_vec, action_space = get_action_space(get_action_limits(get_B(model, trajectory),perc_range), a, num_of_a=3)
    # print(action_space)


    # print(action_space)

    env = LorenzEnvironment(action_space, model, clust_u, n_steps_rk, trajectory)

    if not os.path.exists(filename):
        print('New q-table')
        np.savetxt(filename, init_q_table(env.n_s, env.n_a, init_values))

    a_comp = 3
    q_learning(env, filename, time_filename, save_res, n_episodes, epsilon, alpha, gamma,
               is_rand=is_rand_ic, seed=seed, show_real=False, show_discr=True, a_comp=a_comp,
               null_action=a_null)
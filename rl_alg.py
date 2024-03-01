import os.path
import random
import time

import matplotlib.pyplot as plt
import numpy as np
from s_discretization import random_initial_conditions, states_clustering, get_energy_clust, show_ek, get_laminar_states
from a_discretization import get_action_space, get_action_limits, get_B
from mfe_model import MoehlisFaisstEckhardtModelControl, rk4_timestepping_control

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

class Environment:
    def __init__(self, action_space, m, clust_model, assignments, n_steps):
        self.action_space = action_space
        self.state_space_model = clust_model
        self.n_a = len(action_space)
        self.n_s = len(clust_model.cluster_centers)
        self.n_steps = n_steps

        self.model = m
        self.cur_state = np.zeros(self.model.dim)

        self.start_states = special_ic(clust_model, assignments, model)

        self.delta_t = 0.001
        self.eps = 1e-5
        self.time = 0
        self.T = 100

        self.real_trajectory = np.zeros((self.T+1, self.model.dim))
        self.lam = False

    def set_initial_conditions(self, is_random=True, seed=None):
        if is_random:
            return random_initial_conditions(self.model.dim, seed)
        start_a = np.random.choice(self.start_states)
        # start_a = self.start_states[0]
        return self.state_space_model.cluster_centers[start_a]

    def reset(self, is_rand=True, seed=None):
        self.time = 0
        self.lam = False
        self.real_trajectory = np.zeros((self.T+1, self.model.dim))
        self.cur_state = self.set_initial_conditions(is_rand, seed)
        self.real_trajectory[0] = self.cur_state

        # print('ic:', self.cur_state)
        return self.state_space_model.transform(np.array([self.cur_state]))

    def get_reward(self, action):
        # print(state)
        # print("dist: ", norm(self.cur_state - self.model.laminar_state))
        # print("a: ", norm(action))
        # print(-(norm(self.cur_state - self.model.laminar_state) + 0.5*norm(action)))
        return -norm(self.cur_state - self.model.laminar_state) #- 5*norm(action) #- 0.005*self.time # штраф за т?

    def lam_stop_condition(self, reward):
        return -self.eps <= reward <= self.eps

    def time_stop_condition(self):
        return self.time >= self.T

    def step(self, action_num):
        self.time += self.n_steps * self.delta_t
        if (self.time % 100) == 0:
            print("t = ", self.time)

        prev_state = self.cur_state
        self.cur_state = rk4_timestepping_control(self.model, self.cur_state, self.action_space[action_num],
                                                  delta_t=self.delta_t, n_steps=self.n_steps)[-2]

        next_state = self.state_space_model.transform(np.array([self.cur_state]))
        reward = self.get_reward(self.action_space[action_num])

        self.real_trajectory[int(self.time)] = self.cur_state
        self.da[int(self.time)] = model.f(self.cur_state, self.action_space[action_num])

        if self.time_stop_condition():
            return next_state, reward, True
        elif self.lam_stop_condition(reward):
            print(f'Laminar state, t={env.time}')
            self.lam = True
            return next_state, reward + 10, False
        return next_state, reward, False

    def show_actions(self, actions, a_comp):
        fig, axs = plt.subplots(a_comp, figsize=(10, a_comp*2))
        for i in range(a_comp):
            axs[i].plot(np.arange(len(actions)), [self.action_space[actions[a]][i] for a in range(len(actions))],
                        color='black', markersize=3, linewidth=1)
            axs[i].set(ylabel=f'$a_{i+1}$')
            axs[i].grid()
        plt.xlabel('$t$')
        plt.show()

    def show_trajectory(self, real=False, discr=None):
        if real:
            show_ek([self.model, self.real_trajectory, None, '$E_к$', self.n_steps * self.delta_t], None)
        if discr is not None:
            show_ek(None, [self.model, self.state_space_model, discr, None, '$E_к$', self.n_steps * self.delta_t])





def count_delta(q_n, q_o, t):
    av_d = 0.0
    for i in range(len(q_n)):
        for j in range(len(q_n[0])):
            av_d += abs(q_n[i][j] - q_o[i][j])
    return av_d/(len(q_n)*len(q_n[0])*t)
    # max_d = 0.0
    # for i in range(len(q_n)):
    #     for j in range(len(q_n[0])):
    #         d = abs(q_n[i][j] - q_o[i][j])
    #         if max_d < d:
    #             max_d = d
    # return max_d


def q_learning(env, filename, time_filename, save_res, episodes=1, epsilon=0.15, alpha=0.1, gamma=0.9,
               is_rand=True, seed=None, show_real=False, show_discr=None, a_comp=None, null_action = 0):
    if a_comp is None:
        a_comp=env.model.dim
    q_table = np.loadtxt(filename)

    start_time = time.time()
    for i in range(episodes):
        actions_arr = []
        print("Iteration ", i)
        q_old = np.copy(q_table)
        state = env.reset(is_rand, seed)

        done = False
        states_traj = np.array([state])

        while not done:
            # action = null_action
            if env.time < 40:
                action = null_action
            else:
                # if env.lam:
                #     action = null_action
                    # print(env.action_space[0])
                if random.uniform(0, 1) < epsilon:
                    action = random.randrange(len(env.action_space))
                else:
                    action = np.argmax(q_table[state])

            actions_arr.append(action)

            next_state, reward, done = env.step(action)

            new_value = (1 - alpha) * q_table[state, action] + alpha * (reward + gamma * np.max(q_table[next_state]))
            q_table[state, action] = new_value

            state = next_state
            states_traj = np.append(states_traj, state)
        if save_res:
            # np.savetxt(f'time_series/ep_trajectories_{i}.txt', states_traj, fmt='%1u')
            np.savetxt(filename, q_table)

        print(count_delta(q_table, q_old, env.T))

        # if env.time < env.T:
        #     env.show_trajectory(discr=states_traj)

        if show_discr:
            env.show_trajectory(real=show_real, discr=states_traj)
        else:
            env.show_trajectory(real=show_real)
        # env.show_actions(actions_arr, a_comp)
        env.show_action(actions_arr[int(20/(env.delta_t*env.n_steps)):])



    print("Training finished.\n")
    tt = time.time() - start_time
    print(f"{tt // 60} min, {tt % 60} sec")

    if save_res:
        with open(time_filename, "a") as time_file:
            t_time = str(round(tt/60, 2))
            time_file.write(f"{t_time} {env.T} {episodes}\n")

def init_q_table(n_s, n_a, init_num):
    q = np.full([n_s, n_a], init_num)
    lam_state = env.state_space_model.transform(np.array([env.model.laminar_state]))
    q[lam_state] = np.zeros(n_a)
    return q


if __name__ == "__main__":
    Re = 500.0
    Lx = 1.75 * np.pi
    Lz = 1.2 * np.pi
    model = MoehlisFaisstEckhardtModelControl(Re, Lx, Lz)
    trajectory = np.loadtxt('time_series/trajectory_for_clustering.txt')

    n_states = 800  # число кластеров (дискретных состояний) (>800)
    a = 5  # число действий для одного уравнения
    a_comp = 4  # число уравнений, для которых применяется воздействие
    perc_range = 80  # диапазон значений действий %

    # Параметры алгоритма
    alpha = 0.5
    gamma = 1
    epsilon = 0.2
    n_episodes = 5  # Число эпизодов

    init_values = -1.0  # Начальные значения Q таблицы
    n_steps_rk = 1000  # Число шагов метода Рунге-Кутты (1000)

    is_rand_ic = True  # Случайный выбор начальных состояний
    seed = 6  # Для случайного выбора начальных состояний

    save_res = True
    # Файлы для сохранения результатов
    filename = f'q_tables/q_table_a_{a}_acomp_{a_comp}_rk_{n_steps_rk}_s_{n_states}_perc{perc_range}_continuous.gz'  # + диапазон а
    time_filename = f"simulation_time/time_{n_states}s_{a_comp}comp_{a}a_perc{perc_range}_continuous"

    clust_u, assign_u = states_clustering('kmeans_uniform', trajectory, n_iter_max=1000, n_cl=n_states)
    a_vec, action_space = get_action_space(get_action_limits(get_B(model, trajectory),perc_range), a, num_of_a=a_comp)

    env = Environment(action_space, model, clust_u, assign_u, n_steps_rk)

    if not os.path.exists(filename):
        np.savetxt(filename, init_q_table(env.n_s, env.n_a, init_values))

    q_learning(env, filename, time_filename, save_res, n_episodes, epsilon, alpha, gamma,
               is_rand=is_rand_ic, seed=seed, show_real=False, show_discr=True, a_comp=a_comp)

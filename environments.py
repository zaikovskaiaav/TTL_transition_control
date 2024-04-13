import random
import time
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm

from s_discretization_MFE import random_initial_conditions_mfe, get_energy_clust, show_ek
from MFE_model import MoehlisFaisstEckhardtModelControl, rk4_timestepping_control
from actions_discretization import get_B
from s_discretization_Lorenz import random_initial_conditions_Lorenz
from graphs import show_lorenz_discr_2D, show_lorenz_3D_discr, show_lorenz_2D, show_lorenz_3D


class EnvironmentChaoticSystem:
    def __init__(self, action_space, m, clust_model, assignments, n_steps, T = 100, delta_t = 0.001, eps = 1e-5):
        self.action_space = action_space                # Пространство действий
        self.state_space_model = clust_model            # Модель кластеризации для пространства состояний
        self.n_a = len(action_space)                    # Число действий
        self.n_s = len(clust_model.cluster_centers)     # Число состояний
        self.n_steps = n_steps                          # Число шагов метода Рунгк-Кутты

        self.model = m                                  # Модель системы
        self.cur_state = np.zeros(self.model.dim)       # Текушее состояние

        self.delta_t = delta_t                          # Шаг численного метода
        self.eps = eps                                  # Допуск для устойчивого состояния
        self.time = 0                                   # Время моеделирования
        self.T = T                                      # Время окончания моделирования

        self.real_trajectory = np.zeros((self.T+1, self.model.dim))  # Траектория
        self.lam = False                                # Индикатор устойчивого состояния

    # Задание начальных условий
    def set_initial_conditions(self, is_random=True, seed=None):
        return random_initial_conditions(self.model.dim, seed)

    # Начало нового эпизода
    def reset(self, is_rand=True, seed=None):
        # Сброс параметров
        self.time = 0
        self.lam = False
        self.real_trajectory = np.zeros((self.T+1, self.model.dim))
        self.cur_state = self.set_initial_conditions(is_rand, seed)
        self.real_trajectory[0] = self.cur_state

        return self.state_space_model.transform(np.array([self.cur_state]))

    # Получение награды
    def get_reward(self):
        return -norm(self.cur_state - self.model.laminar_state)

    # Условие остановки при достижении устойчивого состояния
    def lam_stop_condition(self, reward):
        return -self.eps <= reward <= self.eps

    # Условие остановки при достижении конца времени моделирования
    def time_stop_condition(self):
        return self.time >= self.T

    # Шаг алгоритма
    def step(self, action_num):
        # Орпеделение текущего времени
        self.time += self.n_steps * self.delta_t
        if (self.time % 100) == 0:
            print("t = ", self.time)

        # Орпеделение следующего состояния и награды
        prev_state = self.cur_state
        self.cur_state = rk4_timestepping_control(self.model, self.cur_state, self.action_space[action_num],
                                                  delta_t=self.delta_t, n_steps=self.n_steps)[-2]
        next_state = self.state_space_model.transform(np.array([self.cur_state]))
        reward = self.get_reward()

        # Дополнение траектории
        self.real_trajectory[int(self.time)] = self.cur_state

        # Проверка условий остановки
        if self.time_stop_condition():
            return next_state, reward, True
        elif self.lam_stop_condition(reward):
            print(f'Laminar state, t={env.time}')
            self.lam = True
            return next_state, reward + 10, False
        return next_state, reward, False

    # График действий
    def show_actions(self, actions, a_comp):
        fig, axs = plt.subplots(a_comp, figsize=(10, a_comp*2))
        for i in range(a_comp):
            axs[i].plot(np.arange(len(actions)), [self.action_space[actions[a]][i] for a in range(len(actions))],
                        color='black', markersize=3, linewidth=1)
            axs[i].set(ylabel=f'$a_{i+1}$')
            axs[i].grid()
        plt.xlabel('$t$')
        plt.show()

    # График траектории
    def show_trajectory(self, real=False, discr=None):
        if real:
            show_ek([self.model, self.real_trajectory, None, '$E_к$', self.n_steps * self.delta_t], None)
        if discr is not None:
            show_ek(None, [self.model, self.state_space_model, discr, None, '$E_к$', self.n_steps * self.delta_t])



def special_ic(cur_clust, cur_assign, model):
    ke = get_energy_clust(cur_clust, cur_assign, model)
    start_states = []
    start_states_a = []
    for t in range(len(ke)):
        if 19 < ke[t] < 20:
            if cur_assign[t] not in start_states:
                start_states.append(cur_assign[t])
    return start_states



class EnvironmentMFE(EnvironmentChaoticSystem):
    def __init__(self, action_space, m, clust_model, assignments, n_steps, T=100, delta_t=0.001, eps=1e-5):
        super().__init__(action_space, m, clust_model, assignments, n_steps, T, delta_t, eps)
        self.start_states = special_ic(clust_model, assignments, m)

    def set_initial_conditions(self, is_random=True, seed=None):
        if is_random:
            return random_initial_conditions_mfe(self.model.dim, seed)
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
        # self.da[int(self.time)] = model.f(self.cur_state, self.action_space[action_num])

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



class EnvironmentLorenz(EnvironmentChaoticSystem):
    def __init__(self, action_space, m, clust_model, n_steps, traj, assignments):
        super().__init__(action_space, m, clust_model, assignments, n_steps)
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
            return random_initial_conditions_Lorenz(self.model.dim, seed, limit=limit)
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
        self.da[0] = self.model.f(self.cur_state, np.zeros(self.model.dim))
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

        self.da[int(self.time/(self.delta_t*self.n_steps))] = self.model.f(self.cur_state, self.action_space[action_num])
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
            show_lorenz_2D(self.real_trajectory[int(20/(self.delta_t*self.n_steps)):], self.model, isx=True, delta_t=self.delta_t*self.n_steps)
            # show_lorenz_discr_2D(self.model, self.state_space_model, discr)
            show_lorenz_3D_discr(self.model, self.state_space_model, discr)
            if self.lam_t != 0:
                # show_lorenz_3D(self.real_trajectory[:int((self.lam_t) / (self.delta_t * self.n_steps))], self.model)
                # show_lorenz_3D(self.real_trajectory[int((self.lam_t)/(self.delta_t*self.n_steps)):], self.model)
                show_lorenz_3D(self.real_trajectory[:int(40 / (self.delta_t * self.n_steps))], self.model)
                show_lorenz_3D(self.real_trajectory[int(40 / (self.delta_t * self.n_steps)):], self.model)
            else:
                show_lorenz_3D(self.real_trajectory[:int(40 / (self.delta_t * self.n_steps))], self.model)
                show_lorenz_3D(self.real_trajectory[int(40 / (self.delta_t * self.n_steps)):], self.model)
                # show_lorenz_3D(self.real_trajectory, self.model)
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
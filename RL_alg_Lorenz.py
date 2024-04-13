import os.path
import random
import time
import numpy as np

from states_discretization import states_clustering
from s_discretization_Lorenz import random_initial_conditions_Lorenz
from actions_discretization import get_action_space, get_action_limits, get_B
from environments import EnvironmentLorenz
from q_learning_algorithm import q_learning
from Lorenz_model import LorenzModelControl

import matplotlib.pyplot as plt
from numpy.linalg import norm
from copy import copy


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
    ''' Параметры модели '''
    model = LorenzModelControl(10, 28, 8 / 3) # параметры модели

    trajectory = np.loadtxt('time_series/trajectory_for_clustering_lorenz.txt') # траектория для определения пространства состояний
    trajectory_cl = trajectory                                                  # траектория для кластеризации

    ''' Параметры дискретизации '''
    perc_range = 30                                                             # диапазон значений действий %

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

    ''' Параметры дискретизации '''
    n_states = 10           # число кластеров (дискретных состояний)
    a = len(action_space)   # число действий для одного уравнения
    # a = 10
    a_comp = 3              # число уравнений, для которых применяется воздействие


    ''' Параметры алгоритма '''
    alpha = 0.1             # (0.1)
    gamma = 1               # (1)
    epsilon = 0             # (0)

    n_episodes = 10         # Число эпизодов

    init_values = -60       # Начальные значения Q таблицы
    n_steps_rk = 100        # Число шагов метода Рунге-Кутты

    is_rand_ic = True       # Случайный выбор начальных состояний
    seed = None             # Для случайного выбора начальных состояний

    save_res = True

    ''' Сохранение результатов '''
    a_c=3
    filename = f'q_tables/x_left_lorenz_q_table_a{a}_acomp{a_c}_rk{n_steps_rk}_s{n_states}_perc{perc_range}.gz'
    time_filename = f"simulation_time/x_left_lorenz_time_{n_states}s_{a_c}comp_{a}a_perc{perc_range}"

    clust_u, assign_u = states_clustering(trajectory_cl, 'kmeans_uniform', n_iter_max=1000, n_cl=n_states)
    # a_vec, action_space = get_action_space(get_action_limits(get_B(model, trajectory),perc_range), a, num_of_a=3)
    # print(action_space)


    # print(action_space)

    env = EnvironmentLorenz(action_space, model, clust_u, n_steps_rk, trajectory, assign_u)

    if not os.path.exists(filename):
        print('New q-table')
        np.savetxt(filename, init_q_table(env.n_s, env.n_a, init_values))

    a_comp = 3
    q_learning(env, filename, time_filename, save_res, n_episodes, epsilon, alpha, gamma,
               is_rand=is_rand_ic, seed=seed, show_real=False, show_discr=True, a_comp=a_comp,
               null_action=a_null)
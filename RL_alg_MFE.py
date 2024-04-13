import os.path
import random
import time
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm

from states_discretization import states_clustering
from s_discretization_MFE import random_initial_conditions_mfe, get_energy_clust, show_ek, get_laminar_states
from actions_discretization import get_action_space, get_action_limits, get_B
from MFE_model import MoehlisFaisstEckhardtModelControl, rk4_timestepping_control
from environments import EnvironmentMFE
from q_learning_algorithm import q_learning


def init_q_table(n_s, n_a, init_num):
    q = np.full([n_s, n_a], init_num)
    lam_state = env.state_space_model.transform(np.array([env.model.laminar_state]))
    q[lam_state] = np.zeros(n_a)
    return q


if __name__ == "__main__":

    ''' Параметры модели '''
    Re = 500.0
    Lx = 1.75 * np.pi
    Lz = 1.2 * np.pi
    model = MoehlisFaisstEckhardtModelControl(Re, Lx, Lz)
    trajectory = np.loadtxt('time_series/trajectory_for_clustering.txt')

    ''' Параметры дискретизации '''
    n_states = 800          # число кластеров (дискретных состояний) (>800)
    a = 5                   # число действий для одного уравнения (5)
    N_a_comp = 4            # число уравнений, для которых применяется воздействие (4)
    perc_range = 80         # диапазон значений действий % (80)

    ''' Параметры алгоритма '''
    alpha = 0.5             # (0.5)
    gamma = 1               # (1)
    epsilon = 0.2           # (0.2)
    n_episodes = 5          # Число эпизодов (5)
    init_values = -1.0      # Начальные значения Q таблицы (-1.0)

    ''' Параметры моделирования '''
    n_steps_rk = 1000       # Число шагов метода Рунге-Кутты (1000)
    dt = 1 / n_steps_rk     # Шаг метода Рунге-Кутты (0.001)
    T_model = 10            # Время моделирования (100)

    is_rand_ic = True       # Случайный выбор начальных состояний
    seed = 6                # Для случайного выбора начальных состояний

    ''' Файлы для сохранения результатов '''
    save_res = False        # Сохранять Q-table
    filename = f'q_tables/q_table_a_{a}_acomp_{N_a_comp}_rk_{n_steps_rk}_s_{n_states}_perc{perc_range}_continuous.gz'  # + диапазон а
    time_filename = f"simulation_time/time_{n_states}s_{N_a_comp}comp_{a}a_perc{perc_range}_continuous"


    clust_u, assign_u = states_clustering(trajectory, 'kmeans_uniform', n_iter_max=1000, n_cl=n_states)
    a_vec, action_space = get_action_space(get_action_limits(get_B(model, trajectory),perc_range), a, num_of_a=N_a_comp)

    env = EnvironmentMFE(action_space, model, clust_u, assign_u, n_steps_rk, T=T_model, delta_t=dt)

    if save_res and not os.path.exists(filename):
        np.savetxt(filename, init_q_table(env.n_s, env.n_a, init_values))

    q_learning(env, filename, time_filename, save_res, n_episodes, epsilon, alpha, gamma,
               is_rand=is_rand_ic, seed=seed, show_real=False, show_discr=True, a_comp=N_a_comp)

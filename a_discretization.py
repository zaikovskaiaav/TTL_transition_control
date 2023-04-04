import contextvars

from mfe_model import MoehlisFaisstEckhardtModelControl
from s_discretization import generate_trajectory

import numpy as np
import matplotlib.pyplot as plt


# Вычисление правой части системы по известной траектории
def get_B(model, trajectory):
    model.reset_action()
    da = np.zeros_like(trajectory)
    for i in range(len(trajectory)):
        da[i] = model.f(trajectory[i])
    return da

# Вывод графика изменения значений правых частей системы для траектории
def show_B(da):
    da_T = da.T
    for i in range(len(da_T)):
        plt.figure(figsize=(10,3))
        plt.plot(np.arange(len(da)), da_T[i],  linewidth=1, markersize = 0.5)
        plt.xlabel("$t$")
        plt.ylabel(i)
        plt.grid()
        plt.show()

# Вывод распределений для значений правых частей системы
def show_B_distribution(da):
    da_T = da.T
    a_range = np.zeros((len(da_T), 2))

    for i in range(len(da_T)):
        l_perc = np.percentile(da_T[i], 10)
        r_perc = np.percentile(da_T[i], 90)
        a_range[i][0] = l_perc
        a_range[i][1] = r_perc

        step_val = (da_T[i].max() - da_T[i].min())/150
        step = (r_perc - l_perc)/10
    
        count, bins = np.histogram(da_T[i], 150)
        
        plt.figure(figsize=(9,3))
        plt.bar(bins[:-1], count, width = step_val, edgecolor = 'black')
        plt.hist(np.repeat(np.arange(l_perc, r_perc+step, step), count.max()/10),
                 bins = 1, color = 'orange', alpha = 0.3, edgecolor = 'red')
  
        plt.ylim((0, count.max()+300))
        plt.xlabel(f'a_%d' % i)
        plt.grid()
        plt.show()

        # print("Диапазон: ", da_T[i].min(), da_T[i].max())
        # print("Стандартное отклонение: ", np.std(da_T[i]))
        # print("10-й процентиль: ", l_perc)
        # print("90-й процентиль: ", r_perc)

    return a_range


def get_action_limits(da):
    da_T = da.T
    a_range = np.zeros((len(da_T), 2))
    for i in range(len(da_T)):
        l_perc = np.percentile(da_T[i], 10)
        r_perc = np.percentile(da_T[i], 90)
        a_range[i][0] = l_perc
        a_range[i][1] = r_perc
    return a_range

# Генератор случайного управляющего воздействия
def get_action(control_dim, control_range):
    control = np.zeros(control_dim) 
    for i in range(control_dim):
        control[i] = np.random.uniform(control_range[i][0], control_range[i][1])
    return(control)

# Генератор траекторий с постоянным управляющим воздействием
def const_control(model, actions):
    for i in range(len(actions)):
        model.set_action(actions[i])
        test = generate_trajectory(model, time_step, n_steps)
        np.savetxt(f'time_series/ad_test{i}.txt', test)

def get_action_space(a_range, n):
    action_space = np.zeros((len(a_range), n))
    for i in range(len(a_range)):
        action_space[i][0] = 0
        action_space[i][1:] = np.linspace(a_range[i][0], a_range[i][1], n-1)
    #
    # comb_array = np.array(np.meshgrid(action_space[0], action_space[1], action_space[2], action_space[3],
    #                                   action_space[4], action_space[5], action_space[6], action_space[7],
    #                                   action_space[8])).T.reshape(-1, len(a_range))
    comb_array = np.array(np.meshgrid(*action_space)).T.reshape(-1, len(a_range))

    return action_space, comb_array


if __name__ == "__main__":

    Re = 500.0
    Lx = 1.75 * np.pi
    Lz = 1.2 * np.pi

    m = MoehlisFaisstEckhardtModelControl(Re, Lx, Lz)

    time_step = 0.001
    n_steps = 15000000

    # Нахождения значений правых частей системы B и их распределений для готовой траектории
    trajectory = np.loadtxt('time_series/trajectory_for_clustering.txt')

    da = get_B(m, trajectory)
    # show_B(da)
    a_range = show_B_distribution(da)
    print(a_range)

    action_space = get_action_space(a_range, 5)
    print(action_space)

    # # Реализация константного управления
    # actions = np.zeros((6, m.dim))
    # a = [[-0.0001, -0.0005], [0.0001, 0.0005], [0.00005, 0.0002], [-0.00005, -0.0002], [0.00005, 0.00001], [0.00001, 0.0002]]
    # for i in range(len(actions)):
    #     for j in range(len(a[i])):
    #         actions[i][j] = a[i][j]
    #
    # const_control(m, actions)
    #
    # # Вывод распределений правых частей для траекторий с константным управлением
    # for i in range(len(actions)):
    #     trajectory = np.loadtxt(f'time_series/ad_test{i}.txt')
    #     m.reset_action()
    #     da = get_B(m, trajectory)
    #     show_B(da)
    #     show_B_distribution(da)

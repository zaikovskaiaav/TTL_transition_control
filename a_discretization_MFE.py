import contextvars

from mfe_model import MoehlisFaisstEckhardtModelControl
from s_discretization import generate_trajectory

import numpy as np
import matplotlib.pyplot as plt


# Вычисление правой части системы по известной траектории
def get_B(model, trajectory, action=np.zeros(9)):
    da = np.zeros_like(trajectory)
    for i in range(len(trajectory)):
        da[i] = model.f(trajectory[i], action)
    return da

# Вывод графика изменения значений правых частей системы для траектории
def show_B(da, scaling=False):
    da_T = da.T
    p_max = 1
    fig, axs = plt.subplots(len(da_T), figsize=(10, len(da_T) * 2))
    for i in range(len(da_T)):
        if scaling:
            p_max = max(abs(np.min(da_T[i])), np.max(da_T[i]))
        axs[i].plot(np.arange(len(da)), da_T[i]/p_max, 'o--',
                    color='black', linewidth=1, markersize=3)
        axs[i].set(ylabel=f'$p_{i + 1}$')
        axs[i].grid()
    plt.xlabel('$t$')
    plt.show()


# Вывод распределений для значений правых частей системы
def show_B_distribution(da, perc_range):
    lower_perc = (100 - perc_range)/2
    higher_perc = 100 - lower_perc
    da_T = da.T
    a_range = np.zeros((len(da_T), 2))

    labels = ['$x$', '$y$', '$z$']

    for i in range(len(da_T)):
        l_perc = np.percentile(da_T[i], lower_perc)
        r_perc = np.percentile(da_T[i], higher_perc)
        a_range[i][0] = l_perc
        a_range[i][1] = r_perc

        step_val = (da_T[i].max() - da_T[i].min())/70
        step = (r_perc - l_perc)/10
    
        count, bins = np.histogram(da_T[i], 70)
        
        plt.figure(figsize=(7,4))
        plt.bar(bins[:-1], count, width = step_val, edgecolor = 'black', color='gray')
        plt.hist(np.repeat(np.arange(l_perc, r_perc+step, step), count.max()/1),
                     bins = 1, color = 'orange', alpha = 0.3, edgecolor = 'red')

        # plt.hist(np.repeat(np.arange(np.percentile(da_T[i], 25), np.percentile(da_T[i], 75) + step, step), count.max() / 1),
        #          bins=1, color = '#FFAE42', alpha = 0.3, edgecolor = 'red')
        # plt.hist(
        #     np.repeat(np.arange(np.percentile(da_T[i], 35), np.percentile(da_T[i], 65) + step, step), count.max() / 1),
        #     bins=1, color = 'orange', alpha = 0.3, edgecolor = 'red')

  
        plt.ylim((0, count.max()+100))
        plt.xlabel(labels[i])
        plt.ylabel('$N$')
        plt.grid()
        plt.show()

        # print("Диапазон: ", da_T[i].min(), da_T[i].max())
        # print("Стандартное отклонение: ", np.std(da_T[i]))
        # print("10-й процентиль: ", l_perc)
        # print("90-й процентиль: ", r_perc)

    return a_range


def get_action_limits(da, perc_range):
    lower_perc = (100 - perc_range) / 2
    higher_perc = 100 - lower_perc
    da_T = da.T
    a_range = np.zeros((len(da_T), 2))
    for i in range(len(da_T)):
        l_perc = np.percentile(da_T[i], lower_perc)
        r_perc = np.percentile(da_T[i], higher_perc)
        a_range[i][0] = l_perc
        a_range[i][1] = r_perc
    return a_range

# Генератор случайного управляющего воздействия
def get_action(control_dim, control_range):
    control = np.zeros(control_dim) 
    for i in range(control_dim):
        control[i] = np.random.uniform(control_range[i][0], control_range[i][1])
    return(control)

def generate_trajectory_const(model, time_step, n_steps, action, limit=0.2):
    start_time = time.time()
    ic = random_initial_conditions(model.dim, limit=limit)
    trajectory = rk4_timestepping(model, ic, action, time_step, n_steps, time_skip=1000, debug=False)
    print("%s seconds" % (time.time() - start_time))
    return trajectory[:-1]

# Генератор траекторий с постоянным управляющим воздействием
def const_control(model, actions):
    for i in range(len(actions)):
        test = generate_trajectory_const(model, time_step, n_steps, actions[i])
        np.savetxt(f'time_series/ad_test{i}.txt', test)

def get_action_space(a_range, n, num_of_a = 9):
    action_space = np.zeros((num_of_a, n))
    for i in range(num_of_a):
        action_space[i][0] = 0
        action_space[i][1:] = np.linspace(a_range[i][0], a_range[i][1], n-1)

    #
    # comb_array = np.array(np.meshgrid(action_space[0], action_space[1], action_space[2], action_space[3],
    #                                   action_space[4], action_space[5], action_space[6], action_space[7],
    #                                   action_space[8])).T.reshape(-1, len(a_range))

    # comb_array = np.array(np.meshgrid(*action_space)).T.reshape(-1, num_of_a)

    comb_array = action_space[0]
    # print(comb_array)
    for i in range(len(action_space)-1):
        cur_comb = np.empty((len(comb_array)*len(action_space[i+1]), i+2))
        for j in range(len(comb_array)):
            for k in range(len(action_space[i+1])):
                if i == 0:
                    cur_comb[j*len(action_space[i+1])+k] = np.append(np.array([comb_array[j]]), np.array([action_space[i+1][k]]), axis=0)
                else:
                    cur_comb[j * len(action_space[i+1]) + k] = np.append(comb_array[j], np.array([action_space[i + 1][k]]),
                                                      axis=0)
        comb_array = np.copy(cur_comb)

    if num_of_a != len(a_range):
        actions = np.empty((len(comb_array), len(a_range)))
        for i in range(len(comb_array)):
            actions[i] = np.append(comb_array[i], np.zeros(len(a_range)-num_of_a), axis=0)
        return action_space, actions
    return action_space, comb_array


if __name__ == "__main__":

    Re = 500.0
    Lx = 1.75 * np.pi
    Lz = 1.2 * np.pi

    m = MoehlisFaisstEckhardtModelControl(Re, Lx, Lz)

    time_step = 0.001
    n_steps = 15000000

    perc_range = 90

    # Нахождение значений правых частей системы B и их распределений для готовой траектории
    trajectory = np.loadtxt('time_series/trajectory_for_clustering.txt')

    da = get_B(m, trajectory)
    # show_B(da)
    a_range = show_B_distribution(da, perc_range)
    print(a_range)

    action_space, actions = get_action_space(a_range, 5)
    print(action_space)

    # # Реализация константного управления
    # actions = np.zeros((6, m.dim))
    # a = [[-0.0001, -0.0005], [0.0001, 0.0005], [0.00005, 0.0002], [-0.00005, -0.0002], [0.00005, 0.00001], [0.00001, 0.0002]]
    # for i in range(len(actions)):
    #     for j in range(len(a[i])):
    #         actions[i][j] = a[i][j]

    # const_control(m, actions)

    # # Вывод распределений правых частей для траекторий с константным управлением
    # for i in range(len(actions)):
    #     trajectory = np.loadtxt(f'time_series/ad_test{i}.txt')
    #     da = get_B(m, trajectory, actions[i])
    #     show_B(da)
    #     show_B_distribution(da)

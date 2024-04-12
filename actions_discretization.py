import numpy as np
import matplotlib.pyplot as plt

# Вычисление правой части системы по известной траектории
def get_B(model, trajectory, action=np.zeros(9)):
    da = np.zeros_like(trajectory)
    for i in range(len(trajectory)):
        da[i] = model.f(trajectory[i], action)
    return da

# Вывод графика значений правых частей системы для траектории
def show_B(da, scaling=False, ylabel=None, n_dim=9):
    da_T = da.T
    p_max = 1
    fig, axs = plt.subplots(len(da_T), figsize=(10, len(da_T) * 2))
    if len(da_T)  == 3:
        ylabel = ['$x$', '$y$', '$z$']
    for i in range(len(da_T)):
        if scaling:
            p_max = max(abs(np.min(da_T[i])), np.max(da_T[i]))
        axs[i].plot(np.arange(len(da)), da_T[i]/p_max, 'o--',
                    color='black', linewidth=1, markersize=3)
        axs[i].set(ylabel=f'$p_{i + 1}$' if ylabel is None else ylabel[i])
        axs[i].grid()
    plt.xlabel('$t$')
    plt.show()


# Вывод распределений для значений правых частей системы
def get_B_distribution(da, perc_range, ylabel=None, n_dim=9, show=True):
    lower_perc = (100 - perc_range) / 2
    higher_perc = 100 - lower_perc
    da_T = da.T
    a_range = np.zeros((len(da_T), 2))

    if len(da_T)  == 3:
        ylabel = ['$x$', '$y$', '$z$']

    for i in range(len(da_T)):
        l_perc = np.percentile(da_T[i], lower_perc)
        r_perc = np.percentile(da_T[i], higher_perc)
        a_range[i][0] = l_perc
        a_range[i][1] = r_perc
        if show:
            step_val = (da_T[i].max() - da_T[i].min()) / 70
            step = (r_perc - l_perc) / 10

            count, bins = np.histogram(da_T[i], 70)

            plt.figure(figsize=(7, 4))
            plt.bar(bins[:-1], count, width=step_val, edgecolor='black', color='gray')
            plt.hist(np.repeat(np.arange(l_perc, r_perc + step, step), count.max() / 1),
                     bins=1, color='orange', alpha=0.3, edgecolor='red')

            # plt.hist(np.repeat(np.arange(np.percentile(da_T[i], 25), np.percentile(da_T[i], 75) + step, step), count.max() / 1),
            #          bins=1, color = '#FFAE42', alpha = 0.3, edgecolor = 'red')
            # plt.hist(
            #     np.repeat(np.arange(np.percentile(da_T[i], 35), np.percentile(da_T[i], 65) + step, step), count.max() / 1),
            #     bins=1, color = 'orange', alpha = 0.3, edgecolor = 'red')

            plt.ylim((0, count.max() + 100))
            plt.xlabel(f'$a_{i + 1}$' if ylabel is None else ylabel[i])
            plt.ylabel('$N$')
            plt.grid()
            plt.show()

        # print("Диапазон: ", da_T[i].min(), da_T[i].max())
        # print("Стандартное отклонение: ", np.std(da_T[i]))
        # print("10-й процентиль: ", l_perc)
        # print("90-й процентиль: ", r_perc)

    return a_range

# Получение пространства действий в соответствии с заданным диапазоном и количеством
def get_action_space(a_range, n, num_of_a = 9):
    # Определение возможных действий
    action_space = np.zeros((num_of_a, n))
    for i in range(num_of_a):
        action_space[i][0] = 0
        action_space[i][1:] = np.linspace(a_range[i][0], a_range[i][1], n-1)

    #
    # comb_array = np.array(np.meshgrid(action_space[0], action_space[1], action_space[2], action_space[3],
    #                                   action_space[4], action_space[5], action_space[6], action_space[7],
    #                                   action_space[8])).T.reshape(-1, len(a_range))

    # comb_array = np.array(np.meshgrid(*action_space)).T.reshape(-1, num_of_a)

    #
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
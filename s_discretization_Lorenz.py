import matplotlib.pyplot as plt
import numpy as np
from thequickmath.reduced_models.lorenz import LorenzModel
from s_discretization_MFE import states_clustering
from thequickmath.reduced_models.models import rk4_timestepping

import time

#Задание начальных условий, получение траектории
def random_initial_conditions(m_size, seed=None, limit=0.2):
    np.random.seed(seed)
    return np.random.uniform(-limit, limit, size=m_size)

def coord_lorenz(trajectory):
    x = np.zeros(len(trajectory))
    y = np.zeros(len(trajectory))
    z = np.zeros(len(trajectory))
    for i in range(len(trajectory)):
        x[i] = trajectory[i][0]
        y[i] = trajectory[i][1]
        z[i] = trajectory[i][2]
    return x, y, z

def plot_data_lorenz(trajectory, ax, color=None, isx=False):
    x, y, z  = coord_lorenz(trajectory)
    if isx:
        ax.plot3D(x, np.zeros(len(x)), z,  markersize=0.7, color='black')
    else:
        if color:
            ax.plot3D(x, y, z, 'o', markersize=0.7, color=color)
        else:
            ax.plot3D(x, y, z, linewidth=0.7, color='black')

def plot_data_lorenz_2D(trajectory, axs, labels, color=None, n_fig=3, delta_t=0.001):
    x, y, z  = coord_lorenz(trajectory)
    col = 'black'
    t = np.arange(len(trajectory))*(delta_t)
    if color:
        col = color
        axs[0].plot(t, x, color=col, label=labels[0])
        axs[1].plot(t, y, color=col, label=labels[1])
        axs[2].plot(t, z, color=col, label=labels[2])
    else:
        axs[0].plot(t, x, '--', color=col, label=labels[0])
        axs[1].plot(t, y, '--', color=col, label=labels[1])
        axs[2].plot(t, z, '--', color=col, label=labels[2])
    if color:
        for i in range(n_fig):
            axs[i].set(ylabel=labels[i])
            axs[i].grid()
    plt.xlabel(r'$t$')


def plot_x(trajectory, delta_t=0.001):
    x, y, z  = coord_lorenz(trajectory)
    col = 'black'
    t = np.arange(len(trajectory))*(delta_t)
    plt.plot(t, x, color=col, label='x(t)')

    plt.ylabel('$x(t)$')
    plt.grid()
    # plt.xlabel(r'$t$')

def plot_xz(trajectory):
    x, y, z  = coord_lorenz(trajectory)
    col = 'black'

    plt.plot(x, z, color=col, label='x(t)')
    plt.plot(np.sqrt(8 / 3 * 27),  27, 'x', markersize=8, color='gray')
    plt.plot(-np.sqrt(8 / 3 * 27),  27, 'x', markersize=8, color='red')

    plt.ylabel('$z(t)$')
    plt.xlabel('$x(t)$')
    plt.grid()
    # plt.xlabel(r'$t$')


def show_lorenz_2D(trajectory, isx=False, delta_t=0.001, isxz=False):
    n_fig = 3

    labels = ['$x(t)$', '$y(t)$', '$z(t)$']
    if isx:
        plt.figure( figsize=(10, 3))
        plot_x(trajectory, delta_t)
    elif isxz:
        plt.figure(figsize=(7, 4))
        plot_xz(trajectory)
    else:
        fig, axs = plt.subplots(n_fig, figsize=(10, 6))
        plot_data_lorenz_2D(trajectory, axs, labels, color='black', delta_t=delta_t)
    plt.show()

def show_lorenz_discr_2D(model, clust, assign):
    n_fig = 3
    fig, axs = plt.subplots(n_fig, figsize=(10, 6))
    labels = ['$x(t)$', '$y(t)$', '$z(t)$']
    tr_cl = np.zeros((len(assign), model.dim))
    for i in range(len(assign)):
        tr_cl[i] = clust.cluster_centers[assign[i]]
    plot_data_lorenz_2D(tr_cl, axs, labels)
    plt.show()

def show_lorenz_2D_2plots(trajectory, model, clust, assign):
    n_fig = 3
    fig, axs = plt.subplots(n_fig, figsize=(10, 7))
    labels = ['$x(t)$', '$y(t)$', '$z(t)$']

    tr_cl = np.zeros((len(assign), model.dim))
    for i in range(len(assign)):
        tr_cl[i] = clust.cluster_centers[assign[i]]

    plot_data_lorenz_2D(tr_cl, axs, labels)
    plot_data_lorenz_2D(trajectory, axs, labels, color='gray')
    plt.show()


def show_lorenz(trajectory, isx=False):
    # x = np.linspace(-np.pi, np.pi, 50)
    # y = x
    # z = np.cos(x)
    ax = plt.axes(projection='3d')
    # ax.plot3D(0, 0, 0, 'x', markersize=4, color='gray')
    ax.plot3D(np.sqrt(8/3 * 27), np.sqrt(8/3 * 27), 27, 'x', markersize=4, color='gray')
    ax.plot3D(-np.sqrt(8 / 3 * 27), -np.sqrt(8 / 3 * 27), 27, 'x', markersize=4, color='gray')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')
    plot_data_lorenz(trajectory, ax, isx=isx)
    plt.show()

def show_lorenz_discr(model, clust, assign):
    ax = plt.axes(projection ='3d')
    tr_cl = np.zeros((len(assign), model.dim))
    for i in range(len(assign)):
        tr_cl[i] = clust.cluster_centers[assign[i]]
    ax.plot3D(0, 0, 0, 'o', markersize=3, color='red')
    ax.plot3D(np.sqrt(8 / 3 * 27), np.sqrt(8 / 3 * 27), 27, 'o', markersize=3, color='red')
    ax.plot3D(-np.sqrt(8 / 3 * 27), -np.sqrt(8 / 3 * 27), 27, 'o', markersize=3, color='red')
    plot_data_lorenz(tr_cl, ax)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')
    plt.show()

def show_lorenz_2plots(trajectory, model, clust, assign):
    ax = plt.axes(projection ='3d')
    tr_cl = np.zeros((len(assign), model.dim))
    for i in range(len(assign)):
        tr_cl[i] = clust.cluster_centers[assign[i]]

    plot_data_lorenz(trajectory, ax, color='orange')
    plot_data_lorenz(tr_cl, ax,color='#4B0082')
    ax.plot3D(0, 0, 0, 'o', markersize=3, color='red')
    plt.show()

# def gif_lorenz(filename):
#     final = open('togif.txt', 'w')
#     with open(filename, 'r') as file:
#         for i in range(14999):
#             file.readline()
#
#         for i in range(10000):
#             a = file.readline()
#             if i % 50 == 0:
#                 final.write(a+'\n\n')
#     final.close()
#     with open('film.gnu', 'w') as file:
#         file.write('set term gif animate; set output "filmino.gif"' + '\n')
#         for i in range(200):
#             string = f"sp {filename} u 1:2:3 w d,'togif.txt' index " + str(
#                 i) + " u 1:2:3 w p pt 7 ps 3"
#             file.write(string + '\n')

def generate_trajectory_lorenz(model, time_step, n_steps, limit=10):
    start_time = time.time()
    ic = random_initial_conditions(model.dim, limit=limit, is_mfe=False)
    # ic = [14, 14, 49]
    trajectory = rk4_timestepping(model, ic, time_step, n_steps, time_skip=1, debug=False)
    print("%s seconds" % (time.time() - start_time))
    return trajectory[:-1]

# Расчет средней абсолютной процентной погрешности
def calc_mape_lorenz(model, tr, clust, assign):
    mape = 0
    tr_cl = np.zeros((len(assign), model.dim))
    for i in range(len(assign)):
        tr_cl[i] = clust.cluster_centers[assign[i]]
    for i in range(len(tr)):
        mape_cur = 0
        # for j in range(len(tr[0])):
        for j in range(1, 2):
            mape_cur += abs(tr[i][j] - tr_cl[i][j])/abs(tr[i][j])
        mape += mape_cur
    return mape/len(tr) * 100

def mape_of_n_cl_lorenz(n_clust, trajectory, model, tr_test):
    mape_arr = np.zeros_like(n_clust)
    for i in range(len(n_clust)):
        cur_mape = 0
        for j in range(len(tr_test)):
            clust, assign = states_clustering('kmeans_uniform', trajectory, n_iter_max=100000, n_cl=n_clust[i])
            assign_test = clust.transform(tr_test[j])
            cur_mape += calc_mape_lorenz(model, tr_test[j], clust, assign_test)
        mape_arr[i] = cur_mape/len(tr_test)
    plt.plot(n_clust, mape_arr, 'o--', color='black')
    plt.xlabel('$k$')
    plt.ylabel('$MAPE$')
    plt.grid()
    plt.show()

def generate_random_tr(tr):
    tr_T = tr.T
    limits = np.zeros((len(tr_T), 2))
    for i in range(len(tr_T)):
        limits[i] = np.array([np.min(tr_T[i]), np.max(tr_T[i])])
    # print(limits)
    step = 1
    traj = []
    for i in np.arange(limits[0][0], limits[0][1], step):
        for j in np.arange(limits[1][0], limits[1][1], step):
            for k in np.arange(limits[2][0], limits[2][1], step):
                traj.append([i, j, k])

    print(len(traj))
    return np.array(traj)




if __name__ == "__main__":
    get_new_time_series = False # если False, используются сохраненные временные ряды
    time_step = 0.001  # параметры метода Рунге-Кутты
    n_steps = 15000
    n_steps_test = 100000

    do_clustering = False # выполнить кластеризацию
    n_clusters = 10
    do_clustering_analysis =False # вывести зависимость ошибки кластеризации от числа кластеров
    do_msm = False  # выполнить эксперимент с марковским процессом

    if not (do_clustering) and do_msm:
        print(
            "Для построения модели марковского процесса необходимо провести кластеризацию, установите do_clusterng = True")
        sys.exit()

    model = LorenzModel(10, 28, 8 / 3)

    # Получение временных рядов Т=15000
    if get_new_time_series:
        # trajectory = generate_trajectory_lorenz(model, time_step, n_steps, limit=1)
        # np.savetxt('time_series/trajectory_for_clustering_lorenz.txt', trajectory)

        tr_test1 = generate_trajectory_lorenz(model, time_step, n_steps_test, limit=10)
        np.savetxt('time_series/trajectory_test1_lorenz.txt', tr_test1)

        tr_test2 = generate_trajectory_lorenz(model, time_step, n_steps_test, limit=10)
        np.savetxt('time_series/trajectory_test2_lorenz.txt', tr_test2)

        tr_test3 = generate_trajectory_lorenz(model, time_step, n_steps_test, limit=10)
        np.savetxt('time_series/trajectory_test3_lorenz.txt', tr_test3)

        tr_test4 = generate_trajectory_lorenz(model, time_step, n_steps_test, limit=10)
        np.savetxt('time_series/trajectory_test4_lorenz.txt', tr_test4)

    trajectory = np.loadtxt('time_series/trajectory_for_clustering_lorenz.txt')
    tr_test1 = np.loadtxt('time_series/trajectory_test1_lorenz.txt')
    tr_test2 = np.loadtxt('time_series/trajectory_test2_lorenz.txt')
    tr_test3 = np.loadtxt('time_series/trajectory_test3_lorenz.txt')
    tr_test4 = np.loadtxt('time_series/trajectory_test4_lorenz.txt')

    # trajectory = np.loadtxt('time_series/random_trajectory_for_clustering_lorenz.txt')

    # show_lorenz(trajectory)
    # show_lorenz_2D(trajectory)
    # show_lorenz_2D(tr_test1)
    # show_lorenz(tr_test1)
    # show_lorenz_2D(tr_test2)

    show_lorenz_2D(tr_test2, isxz=True)

    # trajectory = generate_random_tr(trajectory)
    # show_lorenz(trajectory)
    # np.savetxt('time_series/random_trajectory_for_clustering_lorenz.txt', trajectory)

    # model = LorenzModel(10, 28, 8/3)
    # trajectory = generate_trajectory_lorenz(model, time_step, 1000000, limit=10)
    # show_lorenz_2D(trajectory)
    # show_lorenz(trajectory)


    # Проведение кластеризации
    if do_clustering:
        clust_u, assign_u = states_clustering(trajectory, 'kmeans_uniform', n_iter_max=1000, n_cl=n_clusters)
        clust_k, assign_k = states_clustering(trajectory, 'kmeans_k++', n_iter_max=1000, n_cl=n_clusters)

        show_lorenz_2D_2plots(trajectory, model, clust_u, assign_u)
        show_lorenz_2plots(trajectory, model, clust_u, assign_u)
        show_lorenz_2D_2plots(trajectory, model, clust_k, assign_k)

        # Вывод тестовых временных рядов
        assign_test1 = clust_u.transform(tr_test1)
        show_lorenz_2D_2plots(tr_test1, model, clust_u, assign_test1)
        show_lorenz_2D(tr_test1)
        show_lorenz(tr_test1)


        assign_test2 = clust_u.transform(tr_test2)
        show_lorenz_2D_2plots(tr_test2, model, clust_u, assign_test2)
        show_lorenz_2D(tr_test2)
        show_lorenz(tr_test2)

        assign_test3 = clust_u.transform(tr_test3)
        show_lorenz_2D_2plots(tr_test3, model, clust_u, assign_test3)
        show_lorenz_2D(tr_test3)
        show_lorenz(tr_test3)

        assign_test4 = clust_u.transform(tr_test4)
        show_lorenz_2D_2plots(tr_test4, model, clust_u, assign_test4)
        show_lorenz_2D(tr_test4)
        show_lorenz(tr_test4)

        # Расчет ошибки кластеризации
        mape = calc_mape_lorenz(model, tr_test1, clust_u, assign_test1)
        print("MAPE:", mape)

    if do_clustering_analysis:
        tr_test_arr = [tr_test1, tr_test2, tr_test3, tr_test4, trajectory]

        n_clust = np.arange(10, 1000, 50)
        mape_of_n_cl_lorenz(n_clust, trajectory, model, tr_test_arr)

        n_clust = np.array([4+i**2 for i in range(10)])
        mape_of_n_cl_lorenz(n_clust, trajectory, model, tr_test_arr)

        # n_clust = np.arange(500, 700, 10)
        # mape_of_n_cl_lorenz(n_clust, trajectory, model, tr_test_arr)

    if do_msm:
        # Нахождение матрицы переходов
        msm = get_msm(assign_u)

        # Нахождение распределение времени жизни турбулентности
        start_time = time.time()
        lam_times = lifetime_distribution(msm, model, clust_u, 1000, 2000)  # 1000, 20000
        print("%s seconds" % (time.time() - start_time))
        show_lifetime_distribution(lam_times)

        # Симуляция марковского процесса
        ic = random_ic_discr_state(model.dim, clust_u)
        msm_t = msm.simulate(15000, ic)
        show_ek(None, [model, clust_u, msm_t, None])

        # Получение распределения на n шагов
        show_distribution_ek(300, model, trajectory, msm, assign_u, clust_u, 0)

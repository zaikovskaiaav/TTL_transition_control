import matplotlib.pyplot as plt
import numpy as np
from thequickmath.reduced_models.lorenz import LorenzModel
from thequickmath.reduced_models.models import rk4_timestepping

from states_discretization import states_clustering
from graphs import show_lorenz_2D, show_lorenz_xyz_2_lines, show_lorenz_clusters_3D, show_lorenz_3D

import time

# Задание начальных условий
def random_initial_conditions_Lorenz(m_size, seed=None, limit=0.2):
    np.random.seed(seed)
    return np.random.uniform(-limit, limit, size=m_size)

# Генерирование траектории для системы Лоренца
def generate_trajectory_lorenz(model, time_step, n_steps, limit=10):
    start_time = time.time()
    ic = random_initial_conditions_Lorenz(model.dim, limit=limit)
    # ic = [14, 14, 49]
    trajectory = rk4_timestepping(model, ic, time_step, n_steps, time_skip=1, debug=False)
    print("Trajectory generation: %s seconds" % (time.time() - start_time))
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
        # for j in range(1, 2):
        j=1
        mape_cur += abs((tr[i][j] - tr_cl[i][j])/tr[i][j])
        mape += mape_cur
    return mape/len(tr) * 100

def mape_of_n_cl_lorenz(n_clust, trajectory, model, tr_test):
    mape_arr = np.zeros_like(n_clust)
    for i in range(len(n_clust)):
        cur_mape = 0
        for j in range(len(tr_test)):
            clust, assign = states_clustering( trajectory, method='kmeans_uniform', n_iter_max=100000, n_cl=n_clust[i])
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
    model = LorenzModel(10, 28, 8 / 3)

    get_new_time_series = False     # Если False, используются сохраненные временные ряды

    time_step = 0.001               # Параметры метода Рунге-Кутты
    n_steps = 15000
    n_steps_test = 100000

    do_clustering = True            # Выполнить кластеризацию
    if do_clustering:
        n_clusters = 10

    do_clustering_analysis = True   # Вывести зависимость ошибки кластеризации от числа кластеров

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

    show_lorenz_2D(tr_test2, model, isxz=True, delta_t=time_step)


    # Проведение кластеризации
    if do_clustering:
        clust_u, assign_u = states_clustering(trajectory, 'kmeans_uniform', n_iter_max=1000, n_cl=n_clusters)
        clust_k, assign_k = states_clustering(trajectory, 'kmeans_k++', n_iter_max=1000, n_cl=n_clusters)

        show_lorenz_xyz_2_lines(trajectory, model, clust_u, assign_u)
        show_lorenz_clusters_3D(trajectory, model, clust_u, assign_u)
        show_lorenz_xyz_2_lines(trajectory, model, clust_k, assign_k)

        # Вывод тестовых временных рядов
        assign_test1 = clust_u.transform(tr_test1)
        show_lorenz_xyz_2_lines(tr_test1, model, clust_u, assign_test1)
        show_lorenz_2D(tr_test1, model)
        show_lorenz_3D(tr_test1, model)

        # show_lorenz_3D_discr(model, clust_u, assign_u)
        # show_lorenz_discr_2D(model, clust_u, assign_u)

        # Расчет ошибки кластеризации
        mape = calc_mape_lorenz(model, tr_test1, clust_u, assign_test1)
        print("MAPE:", mape)

    if do_clustering_analysis:
        tr_test_arr = [tr_test1, tr_test2, tr_test3, tr_test4, trajectory]

        n_clust = np.arange(10, 500, 50)
        mape_of_n_cl_lorenz(n_clust, trajectory, model, tr_test_arr)

        n_clust = np.array([4+i**2 for i in range(10)])
        mape_of_n_cl_lorenz(n_clust, trajectory, model, tr_test_arr)

        # n_clust = np.arange(500, 700, 10)
        # mape_of_n_cl_lorenz(n_clust, trajectory, model, tr_test_arr)



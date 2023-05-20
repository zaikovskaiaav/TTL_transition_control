import matplotlib.pyplot as plt
import numpy as np
from thequickmath.reduced_models.lorenz import LorenzModel
from s_discretization import random_initial_conditions, states_clustering
from thequickmath.reduced_models.models import rk4_timestepping
# from pygnuplot import gnuplot
import time

def plot_data_lorenz(trajectory, ax, color=None):
    x = np.zeros(len(trajectory))
    y = np.zeros(len(trajectory))
    z = np.zeros(len(trajectory))
    for i in range(len(trajectory)):
        x[i] = trajectory[i][0]
        y[i] = trajectory[i][1]
        z[i] = trajectory[i][2]
    if color:
        ax.plot3D(x, y, z, 'o', markersize=0.7, color=color)
    else:
        ax.plot3D(x, y, z, 'o', markersize=0.7)


def show_lorenz(trajectory, model):
    # x = np.linspace(-np.pi, np.pi, 50)
    # y = x
    # z = np.cos(x)
    ax = plt.axes(projection='3d')
    plot_data_lorenz(trajectory, ax)
    ax.plot3D(0, 0, 0, 'o', markersize=3, color='red')
    ax.plot3D(np.sqrt(model.beta * (model.Ra - 1)), np.sqrt(model.beta * (model.Ra - 1)), model.Ra - 1, 'o',
              markersize=3, color='red')
    ax.plot3D(-np.sqrt(model.beta * (model.Ra - 1)), -np.sqrt(model.beta * (model.Ra - 1)), model.Ra - 1, 'o',
              markersize=3, color='red')
    plt.show()

def show_lorenz_discr(model, clust, assign):
    ax = plt.axes(projection ='3d')
    tr_cl = np.zeros((len(assignments), model.dim))
    for i in range(len(assign)):
        tr_cl[i] = clust.cluster_centers[assign[i]]
    plot_data_lorenz(tr_cl, ax)
    ax.plot3D(0, 0, 0, 'o', markersize=3, color='red')
    ax.plot3D(np.sqrt(model.beta*(model.Ra-1)), np.sqrt(model.beta*(model.Ra-1)), model.Ra-1, 'o', markersize=3, color='red')
    ax.plot3D(-np.sqrt(model.beta * (model.Ra - 1)), -np.sqrt(model.beta * (model.Ra - 1)), model.Ra - 1, 'o', markersize=3, color='red')
    plt.show()

def show_lorenz_2plots(trajectory, model, clust, assign):
    ax = plt.axes(projection ='3d')
    tr_cl = np.zeros((len(assign), model.dim))
    for i in range(len(assign)):
        tr_cl[i] = clust.cluster_centers[assign[i]]

    plot_data_lorenz(trajectory, ax, color='orange')
    plot_data_lorenz(tr_cl, ax,color='#4B0082')
    ax.plot3D(0, 0, 0, 'o', markersize=3, color='red')
    ax.plot3D(np.sqrt(model.beta*(model.Ra-1)), np.sqrt(model.beta*(model.Ra-1)), model.Ra-1, 'o', markersize=3, color='red')
    ax.plot3D(-np.sqrt(model.beta * (model.Ra - 1)), -np.sqrt(model.beta * (model.Ra - 1)), model.Ra - 1, 'o', markersize=3, color='red')
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

def generate_trajectory_lorenz(model, time_step, n_steps, limit=0.2):
    start_time = time.time()
    ic = random_initial_conditions(model.dim, limit=limit, is_mfe=False)
    # ic = [10, 10, 10]
    trajectory = rk4_timestepping(model, ic, time_step, n_steps, time_skip=1000, debug=False)
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
        for j in range(1):
            mape_cur += abs(tr[i][j] - tr_cl[i][j])/abs(tr[i][j])
        mape += mape_cur/3
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
    plt.plot(n_clust, mape_arr, 'o--')
    plt.xlabel('$n$')
    plt.ylabel('$MAPE$')
    plt.grid()
    plt.show()


if __name__ == "__main__":
    get_new_time_series = False  # если False, используются сохраненные временные ряды
    time_step = 0.001  # параметры метода Рунге-Кутты
    n_steps = 15000000

    do_clustering = True  # выполнить кластеризацию
    n_clusters = 1000
    do_clustering_analysis = True  # вывести зависимость ошибки кластеризации от числа кластеров
    do_msm = False  # выполнить эксперимент с марковским процессом

    if not (do_clustering) and do_msm:
        print(
            "Для построения модели марковского процесса необходимо провести кластеризацию, установите do_clusterng = True")
        sys.exit()

    model = LorenzModel(10, 28, 8 / 3)

    # Получение временных рядов Т=15000
    if get_new_time_series:
        trajectory = generate_trajectory_lorenz(model, time_step, n_steps, limit=10)
        np.savetxt('time_series/trajectory_for_clustering_lorenz.txt', trajectory)

        tr_test1 = generate_trajectory_lorenz(model, time_step, n_steps, limit=10)
        np.savetxt('time_series/trajectory_test1_lorenz.txt', tr_test1)

        tr_test2 = generate_trajectory_lorenz(model, time_step, n_steps, limit=10)
        np.savetxt('time_series/trajectory_test2_lorenz.txt', tr_test2)

    trajectory = np.loadtxt('time_series/trajectory_for_clustering_lorenz.txt')
    tr_test1 = np.loadtxt('time_series/trajectory_test1_lorenz.txt')
    tr_test2 = np.loadtxt('time_series/trajectory_test2_lorenz.txt')

    # gif_lorenz('time_series/trajectory_for_clustering_lorenz.txt')
    # show_lorenz(trajectory, model)
    # show_lorenz(tr_test1, model)
    # show_lorenz(tr_test2, model)

    # Проведение кластеризации
    if do_clustering:
        clust_u, assign_u = states_clustering('kmeans_uniform', trajectory, n_iter_max=1000, n_cl=n_clusters)
        clust_k, assign_k = states_clustering('kmeans_k++', trajectory, n_iter_max=1000, n_cl=n_clusters)

        show_lorenz_2plots(trajectory, model, clust_u, assign_u)
        show_lorenz_2plots(trajectory, model, clust_k, assign_k)

        # Вывод тестовых временных рядов
        assign_test1 = clust_u.transform(tr_test1)
        show_lorenz_2plots(trajectory, model, clust_u, assign_test1)

        assign_test2 = clust_u.transform(tr_test2)
        show_lorenz_2plots(trajectory, model, clust_u, assign_test2)

        # Расчет ошибки кластеризации
        mape = calc_mape_lorenz(model, tr_test1, clust_u, assign_test1)
        print("MAPE:", mape)

    if do_clustering_analysis:
        tr_test_arr = [tr_test1, tr_test2, trajectory]

        n_clust = np.array([100, 500, 1000, 2000, 3000, 5000, 8000, 10000])
        mape_of_n_cl_lorenz(n_clust, trajectory, model, tr_test_arr)

        n_clust = np.arange(50, 1000, 50)
        mape_of_n_cl_lorenz(n_clust, trajectory, model, tr_test_arr)

        n_clust = np.arange(500, 700, 10)
        mape_of_n_cl_lorenz(n_clust, trajectory, model, tr_test_arr)

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

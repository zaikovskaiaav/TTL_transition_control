# from mfe_model import rk4_timestepping_control
from thequickmath.reduced_models.transition_to_turbulence import MoehlisFaisstEckhardtModel
from thequickmath.reduced_models.models import rk4_timestepping
from states_discretization import states_clustering, calc_mape, mape_of_n_cl
from graphs import show_ek

from deeptime.clustering import KMeans
import deeptime.markov as markov
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import time
import sys

# Задание начальных условий
def random_initial_conditions_mfe(m_size, seed=None, limit=0.2):
    np.random.seed(seed)
    ic = np.random.uniform(-limit, limit, size=m_size)
    ic[0] = np.random.uniform(0, limit)
    ic[-1] = np.random.uniform(-limit, 0)
    return ic

# Получение траектории
def generate_trajectory(model, time_step, n_steps, limit=0.5):
    start_time = time.time()
    ic = random_initial_conditions(model.dim, limit=limit)
    trajectory = rk4_timestepping(model, ic, time_step, n_steps, time_skip=1000, debug=False)
    print("Trajectory generation: %s seconds" % (time.time() - start_time))
    return trajectory[:-1]

# Получение списка ламинарных состояний
def get_laminar_states(cur_clust, cur_assign, model):
    ke = get_energy_clust(cur_clust, cur_assign, model)
    lam_states = []
    lam_states_a = []
    for t in range(len(ke)):
        if ke[t] > 20:
            if cur_assign[t] not in lam_states:
                # Сохранение всех ламинарных состояний в порядке роста их Е (в порядке появления во временном ряде)
                lam_states.append(cur_assign[t])
                lam_states_a.append(cur_clust.cluster_centers[cur_assign[t]])
    return lam_states_a


'''         Эксперимент с MSM            '''

# Получение кинетической энергии для дискретной траектории
def get_energy_clust(clustering, assignments, model):
    tr_cl = np.zeros((len(assignments), model.dim))
    for i in range(len(assignments)):
        if assignments[i] == -1:
            tr_cl[i] = tr_cl[i-1]
        else:
            tr_cl[i] = clustering.cluster_centers[assignments[i]]
    return model.kinetic_energy(tr_cl)

# Получение модели марковского процесса
def get_msm(cur_assign):
    estimator_mlm = markov.msm.MaximumLikelihoodMSM(
        reversible=False,
        stationary_distribution_constraint=None
    #     transition_matrix_tolerance = 1e-5
    )
    start_time = time.time()
    msm = estimator_mlm.fit(cur_assign, lagtime=1).fetch_model()  # MSM для дискретной траектории
    print("MSM: %s seconds" % (time.time() - start_time))
    return msm

# Генерация случайных н.у. в виде дискретного состояния
def random_ic_discr_state_mfe(m_size, clust):
    ic = random_initial_conditions_mfe(m_size)
    return clust.transform(np.array([ic]))

# Получение траекторий для симуляции марковского процесса
def msm_simulation(msm, m_size, time, clust):
    return msm.simulate(time, int(random_ic_discr_state_mfe(m_size, clust)[0]))

# Получение функции выживаемости
def survival_function(data):
    values = sorted([t for t in data if t != None])
    probs = np.array([1 - i/len(values) for i in range(len(values))])
    return values, probs

# Время жизни турбулентности
def relaminarisation_time(ke, T=1000):
    transition_start = None
    for t in range(len(ke)):
        if transition_start:
            if t - transition_start > T:
                return t
            if ke[t] < 20:
                transition_start = None
        elif ke[t] > 20:
            transition_start = t
    last_t = len(ke) - 1
    return last_t if transition_start is not None else None

# Получение распределения времени жизни
def lifetime_distribution(msm, model, clust, n, time):
    T = 1000
    lam_times = []
    for i in range(n):
        tr = msm_simulation(msm, model.dim, time, clust)
        ke = get_energy_clust(clust, tr, model)
        lam_times.append(relaminarisation_time(ke, T))
    return lam_times

# Вывод распределения времени жизни турбулентности
def show_lifetime_distribution(lam_times):
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    sf = survival_function(lam_times)
    lines = ax.semilogy(sf[0], sf[1], 'o--', color='black', linewidth=1.5, markersize=3.5)
    ax.grid()
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$S(t)$')
    plt.show()


# Получение кинетической энергии каждого состояния траектории
def get_ek_trajectory(model, trajectory, tr_assignments):
    ek_states = np.zeros(len(tr_assignments))
    ek_trajectory = model.kinetic_energy(trajectory)
    for i in range(len(tr_assignments)):
        ek_states[tr_assignments[i]] = ek_trajectory[i]
    return ek_states, ek_trajectory


# Математическое ожидание
def calc_expectation(p, a):
    ex = 0
    for i in range(len(p)):
        ex += p[i] * a[i]
    return ex

# Нахождение распределения вероятностей значений кинетической энергии
def get_distribution_ek(n, model, trajectory, msm, tr_assignments, start):
    expectation = np.zeros(n) # Математическое ожидание
    ek_states, ek_trajectory = get_ek_trajectory(model, trajectory, tr_assignments)

    p0 = np.zeros(len(msm.transition_matrix)) # Начальное распределение
    p0[tr_assignments[start]] = 1.

    p = np.zeros((n, len(msm.transition_matrix))) # Массив распределений вероятности

    ek_i = np.zeros((n+1, int(np.max(ek_states))+1)) # Массив вероятностей, соответствующих уровням кинетической энергии
    ek_i[0][int(ek_states[tr_assignments[start]])] = p0[tr_assignments[start]]

    p[0] = p0
    for i in range(n):
        p0 = msm.propagate(p0, 1)
        p[i] = p0
        for s in range(len(p0)):
            if p0[s] != 0:
                ek_i[i+1][int(ek_states[s])] += p0[s]
        expectation[i] = calc_expectation(p0, ek_states)
    return ek_i, expectation, ek_trajectory


# График распределения кинетической энергии
def show_distribution_ek(n, model, trajectory, msm, tr_assignments, clustering, start):

    ek_i, expectation, ek_trajectory = get_distribution_ek(n, model, trajectory, msm, tr_assignments, start)

    fig, ax = plt.subplots(figsize=(10, 8))

    tr_clust = np.zeros((len(tr_assignments), model.dim))
    for i in range(len(tr_assignments)):
        tr_clust[i] = clustering.cluster_centers[tr_assignments[i]]

    ek_tr = model.kinetic_energy(tr_clust[start:n+start])

    # ax.plot(np.arange(len(tr_clust[start:n+start])), ek_tr, "b--", linewidth=1, label = f'Дискретная траектория')
    ax.plot(np.arange(len(trajectory[start:n+start])), ek_trajectory[start:n+start], "black", linewidth=1, label = "Исходная траектория")
    ax.plot(np.arange(n), expectation, 'k--', linewidth=1, label = "Математическое ожидание")

    ax.set_xlabel('$t$')
    ax.set_ylabel('$E$')

    p1 = ax.imshow(np.transpose(ek_i), origin='lower', aspect='auto', cmap='Oranges', alpha=0.9)
    # plt.legend()

    # axins = inset_axes(ax,
    #                    width="5%",
    #                    height="70%",
    #                    loc='lower left',
    #                    bbox_to_anchor=(1.05, 0, 1, 1),
    #                    # bbox_to_anchor=(1, 0, 1, 1),
    #                    bbox_transform=ax.transAxes,
    #                    borderpad=0,
    #                    )
    # fig.colorbar(p1, cax=axins, label='Вероятность')
    fig.colorbar(p1, ax=ax, orientation='horizontal', pad=0.1, label='Вероятность', shrink=0.6  )
    # fig.tight_layout()
    plt.subplot_tool()
    plt.show()


if __name__ == "__main__":
    # Параметры системы
    Re = 500.0
    Lx = 1.75 * np.pi
    Lz = 1.2 * np.pi
    model = MoehlisFaisstEckhardtModel(Re, Lx, Lz)

    get_new_time_series = False     # Если False, используются сохраненные временные ряды
    if get_new_time_series:
        # Параметры метода Рунге-Кутты
        time_step = 0.001
        n_steps = 15000000

    do_clustering = True            # Выполнить кластеризацию
    if do_clustering:
        n_clusters = 850

    do_clustering_analysis = False  # Вывести зависимость ошибки кластеризации от числа кластеров

    do_msm = True                   # Выполнить эксперимент с марковским процессом

    if not(do_clustering) and do_msm:
        print("Для построения модели марковского процесса необходимо провести кластеризацию, установите do_clusterng = True")
        sys.exit()

    # Получение временных рядов Т=15000
    if get_new_time_series:
        trajectory = generate_trajectory(model, time_step, n_steps)
        np.savetxt('time_series/trajectory_for_clustering.txt', trajectory)

        tr_test1 = generate_trajectory(model, time_step, n_steps)
        np.savetxt('time_series/trajectory_test1.txt', tr_test1)

        tr_test2 = generate_trajectory(model, time_step, n_steps)
        np.savetxt('time_series/trajectory_test2.txt', tr_test2)

    trajectory = np.loadtxt('time_series/trajectory_for_clustering.txt')
    tr_test1 = np.loadtxt('time_series/trajectory_test1.txt')
    tr_test2 = np.loadtxt('time_series/trajectory_test2.txt')


    # Проведение кластеризации
    if do_clustering:
        clust_u, assign_u = states_clustering(trajectory, 'kmeans_uniform', n_iter_max = 1000, n_cl = n_clusters)
        clust_k, assign_k = states_clustering(trajectory, 'kmeans_k++', n_iter_max = 1000, n_cl = n_clusters)

        # Вывод графиков кинетической энергии для непрерывной и дискретной траектории
        show_ek([model, trajectory, None], [model, clust_u, assign_u, None])
        show_ek([model, trajectory, None], [model, clust_k, assign_k, None])

        # Вывод тестовых временных рядов
        assign_test1 = clust_u.transform(tr_test1)
        show_ek([model, tr_test1, None], [model, clust_u, assign_test1, None])
        assign_test2 = clust_u.transform(tr_test2)
        show_ek([model, tr_test2, None], [model, clust_u, assign_test2, None])

        # Расчет ошибки кластеризации
        mape = calc_mape(model, tr_test1, clust_u, assign_test1)
        print("MAPE = ", mape)

    if do_clustering_analysis:
        # tr_test5 = generate_trajectory(model, time_step, n_steps, limit=1)
        # np.savetxt('time_series/trajectory_test5.txt', tr_test5)
        # tr_test6 = generate_trajectory(model, time_step, n_steps, limit=1)
        # np.savetxt('time_series/trajectory_test6.txt', tr_test6)

        tr_test3 = np.loadtxt('time_series/trajectory_test3.txt')
        tr_test4 = np.loadtxt('time_series/trajectory_test4.txt')
        tr_test5 = np.loadtxt('time_series/trajectory_test5.txt')
        tr_test6 = np.loadtxt('time_series/trajectory_test6.txt')

        show_ek([model, tr_test5, None], None)
        show_ek([model, tr_test6, None], None)

        tr_test_arr = [tr_test1, tr_test2, tr_test3, tr_test4, tr_test5, tr_test6, trajectory]

        # Расчет ошибки кластеризации для разного числа кластеров
        n_clust = np.array([100, 500, 1000, 2000, 3000, 5000, 8000, 10000])
        mape_of_n_cl(n_clust, trajectory, model, tr_test_arr)

        n_clust = np.arange(50, 1000, 50)
        mape_of_n_cl(n_clust, trajectory, model, tr_test_arr)

        n_clust = np.arange(500, 700, 10)
        mape_of_n_cl(n_clust, trajectory, model, tr_test_arr)

    if do_msm:
        #Нахождение матрицы переходов
        msm = get_msm(assign_u)

        # Нахождение распределение времени жизни турбулентности
        start_time = time.time()
        lam_times = lifetime_distribution(msm, model, clust_u, 1000, 200)           #1000, 20000
        print("Lifetime distribution: %s seconds" % (time.time() - start_time))
        show_lifetime_distribution(lam_times)

        # Симуляция марковского процесса
        ic = random_ic_discr_state_mfe(model.dim, clust_u)
        msm_t = msm.simulate(15000, ic)
        show_ek(None, [model, clust_u, msm_t, 'black'])

        # Получение распределения на n шагов
        show_distribution_ek(1000, model, trajectory, msm, assign_u, clust_u, 0)    # n=15000


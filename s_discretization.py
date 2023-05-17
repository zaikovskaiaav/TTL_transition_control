from mfe_model import rk4_timestepping_control
from thequickmath.reduced_models.transition_to_turbulence import MoehlisFaisstEckhardtModel
# from thequickmath.reduced_models.models import rk4_timestepping

from deeptime.clustering import KMeans
import deeptime.markov as markov
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import time
import sys


# class Clustering:
#     def __init__(self, trajectory, method, n_iter_max=1000, n_cl=500):
#         self.trajectory = trajectory
#         if method == 'kmeans_uniform':
#             init_c = 'uniform'
#         else:
#             init_c = 'kmeans++'
#         estimator = KMeans(
#             n_clusters=n_cl,
#             init_strategy=init_c,
#             max_iter=n_iter_max,
#             fixed_seed=13,
#             n_jobs=8)
#
#         start_time = time.time()
#         self.clustering = estimator.fit(trajectory).fetch_model()
#         self.assignments = self.clustering.transform(trajectory)
#         print("%s seconds" % (time.time() - start_time))
#
#     def get_energy_clust(self, model, assignments):
#         tr_cl = np.zeros((len(assignments), model.dim))
#         for i in range(len(assignments)):
#             if assignments[i] == -1:
#                 tr_cl[i] = tr_cl[i - 1]
#             else:
#                 tr_cl[i] = self.clustering.cluster_centers[assignments[i]]
#         return model.kinetic_energy(tr_cl)
#
#     def show_energy_clust(self, model, traj, assignments, label):
#         ek = self.get_energy_clust(model, assignments)
#         if label:
#             plt.plot(np.arange(len(traj)), ek, linewidth=0.7, color=label, markersize=0.5)
#         else:
#             plt.plot(np.arange(len(traj)), ek, linewidth=1, markersize=0.5)
#         plt.xlabel("$t$")
#         plt.ylabel("$E$")
#         return ek

    # def show_inertia(self, clust1, clust2):
    #     plt.figure(figsize=(5, 3))
    #     plt.loglog(clust1.inertias, label='cl1')
    #     plt.loglog(clust2.inertias, label='cl2')
    #     plt.grid()
    #     plt.legend()
    #     plt.xlabel("iteration")
    #     plt.ylabel("inertia")



#Задание начальных условий, получение траектории

def random_initial_conditions(m_size, seed=None):
    np.random.seed(seed)
    limit = 0.2
    ic = np.random.uniform(-limit, limit, size=m_size)
    ic[0] = np.random.uniform(0, limit)
    ic[-1] = np.random.uniform(-limit, 0)
    return ic

def generate_trajectory(model, time_step, n_steps, action=np.zeros(9)):
    start_time = time.time()
    ic = random_initial_conditions(model.dim)
    trajectory = rk4_timestepping_control(model, ic, action, time_step, n_steps, time_skip=1000, debug=False)
    print("%s seconds" % (time.time() - start_time))
    return trajectory[:-1]


# Вывод графиков

def print_2d(v):
    fig, ax = plt.subplots(1, 1, figsize=(5,3), constrained_layout=True)
   
    p1 = ax.imshow(v, cmap='viridis', aspect='equal', origin="lower")
    ax.set_title("Flow")

    axins = inset_axes(ax,
                   width="7%",
                   height="70%",
                   loc='lower left',
                   bbox_to_anchor=(1.05, 0., 1, 1),
                   bbox_transform=ax.transAxes,
                   )
    plt.colorbar(p1, ax=ax, cax=axins, label='Velocity')
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    
def show_flow(trajectory, y, l):
    points = np.arange(0, len(trajectory), len(trajectory)//l)
    d_x = np.arange(0, Lx, 0.05)
    d_z = np.arange(0, Lz, 0.05)
    field_y0 = np.zeros((len(points), len(d_z), len(d_x)))
    for i in range(len(points)):
        for x in range(len(d_x)):
            for z in range(len(d_z)):
                field_y0[i][z][x] = model.three_dim_flow_field(trajectory[points[i]], d_x[x], y, d_z[z])
    return field_y0

def show_energy(model, trajectory, color, ylab="$E$", k=1):
    ek = model.kinetic_energy(trajectory)
    if color:
        plt.plot(np.arange(len(ek))*k, ek, linewidth=0.7, color=color, markersize = 0.5)
    else:
        plt.plot(np.arange(len(ek))*k, ek,  linewidth=1, markersize = 0.5)
    plt.xlabel("$t$")
    plt.ylabel(ylab)
    # return ek


# Кластеризация

def states_clustering(method, trajectory, n_iter_max = 1000, n_cl = 500, dist = 3):
    if method == 'kmeans_uniform':
        init_c = 'uniform'
    else:
        init_c = 'kmeans++'
    estimator = KMeans(
        n_clusters = n_cl,
        init_strategy = init_c,
        max_iter = n_iter_max,
        fixed_seed=13,
        n_jobs=8)

    start_time = time.time()
    clustering = estimator.fit(trajectory).fetch_model()
    assignments = clustering.transform(trajectory)
    print("%s seconds" % (time.time() - start_time))
    return clustering, assignments


def get_energy_clust(clustering, assignments, model):
    tr_cl = np.zeros((len(assignments), model.dim))
    for i in range(len(assignments)):
        if assignments[i] == -1:
            tr_cl[i] = tr_cl[i-1]
        else:
            tr_cl[i] = clustering.cluster_centers[assignments[i]]
    return model.kinetic_energy(tr_cl)

# def show_energy_clust(model, clustering, assignments, label):
#     ek = get_energy_clust(clustering, assignments, model)
#     if label:
#         plt.plot(np.arange(len(ek)), ek, linewidth=0.7, color = label, markersize = 0.5)
#     else:
#         plt.plot(np.arange(len(ek)), ek,  linewidth=1, markersize = 0.5)
#     plt.xlabel("$t$")
#     plt.ylabel("$E$")
#     # return ek

def show_energy_clust(model, clustering, assignments, color, ylab="$E$", k=1):
    tr_cl = np.zeros((len(assignments), model.dim))
    for i in range(len(assignments)):
        if assignments[i] == -1:
            tr_cl[i] = tr_cl[i - 1]
        else:
            tr_cl[i] = clustering.cluster_centers[assignments[i]]
    show_energy(model, tr_cl, color, ylab, k)

# def show_inertia(clust1, clust2):
#     plt.figure(figsize = (5,3))
#     plt.loglog(clust1.inertias, label='cl1')
#     plt.loglog(clust2.inertias, label='cl2')
#     plt.grid()
#     plt.legend()
#     plt.xlabel("iteration")
#     plt.ylabel("inertia")


# Расчет средней абсолютной процентной погрешности
def calc_mape(model, tr, clust, assign):
    mape = 0
    ek_tr = model.kinetic_energy(tr)
    tr_cl = np.zeros((len(assign), model.dim))
    for i in range(len(assign)):
        if assign[i] == -1:
            tr_cl[i] = tr_cl[i-1] 
        else:
            tr_cl[i] = clust.cluster_centers[assign[i]] 
    ek_discr = model.kinetic_energy(tr_cl)        

    for j in range(len(tr)):
        mape += abs(ek_tr[j] - ek_discr[j])/abs(ek_tr[j])
    return mape/len(tr) * 100

def get_msm(cur_assign):
    estimator_mlm = markov.msm.MaximumLikelihoodMSM(
        reversible=False,
        stationary_distribution_constraint=None
    #     transition_matrix_tolerance = 1e-5
    )
    start_time = time.time()
    msm = estimator_mlm.fit(cur_assign, lagtime=1).fetch_model()  # MSM для дискретной траектории
    print("%s seconds" % (time.time() - start_time))
    return msm


# Генерация случайных н.у. в виде дискретного состояния
def random_ic_discr_state(m_size, clust):
    ic = random_initial_conditions(m_size)
    return clust.transform(np.array([ic]))

# Получение траекторий 
def msm_simulation(msm, m_size, time, clust):
    return msm.simulate(time, int(random_ic_discr_state(m_size, clust)[0]))


def survival_function(data, debug=False):
    values = sorted([t for t in data if t != None])
    if debug and len(data) != len(values):
        print(f'While building survival function, filtered {len(data) - len(values)} "None" points')
    probs = np.array([1 - i/len(values) for i in range(len(values))])
    return values, probs

def relaminarisation_time(ke, T=1000, debug=False):
    '''
    We detect turbulent-to-laminar transition if the kinetic energy is larger than 15 for more than T time units
    and return relaminarisation time is this event has occured. Otherwise None is returned
    '''
    transition_start = None
    for t in range(len(ke)):
        if transition_start:
            if t - transition_start > T:
                if debug:
                    print(f'Found turbulent-to-laminar transition from {transition_start} to {t}')
                return t
            if ke[t] < 20:
                transition_start = None
        elif ke[t] > 20:
            transition_start = t
    last_t = len(ke) - 1
    if debug and transition_start is not None:
        print(f'Found turbulent-to-laminar transition from {transition_start} to infty ({last_t})')
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


def show_lifetime_distribution(lam_times):
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    sf = survival_function(lam_times, True)
    lines = ax.semilogy(sf[0], sf[1], 'o--', linewidth=1.5, markersize=3.5)

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

    ek_i = np.zeros((n+1, int(ek_states.max())+1)) # Массив вероятностей, соответствующих уровням кинетической энергии
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


def show_distribution_ek(n, model, trajectory, msm, tr_assignments, clustering, start):
    
    ek_i, expectation, ek_trajectory = get_distribution_ek(n, model, trajectory, msm, tr_assignments, start)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    tr_clust = np.zeros((len(tr_assignments), model.dim))
    for i in range(len(tr_assignments)): 
        tr_clust[i] = clustering.cluster_centers[tr_assignments[i]]

    ek_tr = model.kinetic_energy(tr_clust[start:n+start])

    ax.plot(np.arange(len(tr_clust[start:n+start])), ek_tr, "b--", linewidth=1, label = f'Дискретная траектория')
    ax.plot(np.arange(len(trajectory[start:n+start])), ek_trajectory[start:n+start], "#00008B", linewidth=1, label = "Исходная траектория")
    ax.plot(np.arange(n), expectation, '#800000', linewidth=1, label = "Математическое ожидание")

    ax.set_xlabel('$t$')
    ax.set_ylabel('$E$')

    p1 = ax.imshow(np.transpose(ek_i), origin='lower', aspect='auto', cmap='Reds', alpha=0.9)
    plt.legend()

    axins = inset_axes(ax,
                       width="7%",
                       height="70%",
                       loc='lower left',
                       bbox_to_anchor=(1.05, 0., 1, 1),
                       bbox_transform=ax.transAxes,
                       )
    plt.colorbar(p1, ax=ax, cax=axins, label='Вероятность')
    plt.show()


def show_ek(tr, discr):
    plt.figure(figsize=(10, 3))
    if tr and discr:
        show_energy_clust(*discr)
        show_energy(*tr)
    elif tr:
        show_energy(*tr)
    else:
        show_energy_clust(*discr)
    plt.grid()
    plt.show()

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

def mape_of_n_cl(n_clust, trajectory, model, tr_test):
    mape_arr = np.zeros_like(n_clust)
    for i in range(len(n_clust)):
        cur_mape = 0
        for j in range(len(tr_test)):
            clust, assign = states_clustering('kmeans_uniform', trajectory, n_iter_max=100000, n_cl=n_clust[i])
            assign_test = clust.transform(tr_test[j])
            cur_mape += calc_mape(model, tr_test[j], clust, assign_test)
        mape_arr[i] = cur_mape/len(tr_test)
    plt.plot(n_clust, mape_arr, 'o--')
    plt.xlabel('$n$')
    plt.ylabel('$MAPE$')
    plt.grid()
    plt.show()


if __name__ == "__main__":

    get_new_time_series = False  # если False, используются сохраненные временные ряды
    time_step = 0.001   # параметры метода Рунге-Кутты
    n_steps = 15000000

    do_clustering = True  # выполнить кластеризацию
    n_clusters = 850
    do_clustering_analysis = True  # вывести зависимость ошибки кластеризации от числа кластеров
    do_msm = False  # выполнить эксперимент с марковским процессом

    if not(do_clustering) and do_msm:
        print("Для построения модели марковского процесса необходимо провести кластеризацию, установите do_clusterng = True")
        sys.exit()

    Re = 500.0
    Lx = 1.75 * np.pi
    Lz = 1.2 * np.pi

    model = MoehlisFaisstEckhardtModel(Re, Lx, Lz)

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
        clust_u, assign_u = states_clustering('kmeans_uniform', trajectory, n_iter_max = 1000, n_cl = n_clusters)
        clust_k, assign_k = states_clustering('kmeans_k++', trajectory, n_iter_max = 1000, n_cl = n_clusters)

        show_ek([model, trajectory, '#4B0082'], [model, clust_u, assign_u, 'orange'])
        show_ek([model, trajectory, '#4B0082'], [model, clust_k, assign_k, 'orange'])

        # Вывод тестовых временных рядов
        assign_test1 = clust_u.transform(tr_test1)
        show_ek([model, tr_test1, '#4B0082'], [model, clust_u, assign_test1, 'orange'])

        assign_test2 = clust_u.transform(tr_test2)
        show_ek([model, tr_test2, '#4B0082'], [model, clust_u, assign_test2, 'orange'])

        # Расчет ошибки кластеризации
        mape = calc_mape(model, tr_test1, clust_u, assign_test1)
        print("MAPE:", mape)


    if do_clustering_analysis:
        tr_test_arr = [tr_test1, tr_test2, trajectory]

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
        lam_times = lifetime_distribution(msm, model, clust_u, 1000, 2000) #1000, 20000
        print("%s seconds" % (time.time() - start_time))
        show_lifetime_distribution(lam_times)

        # Симуляция марковского процесса
        ic = random_ic_discr_state(model.dim, clust_u)
        msm_t = msm.simulate(15000, ic)
        show_ek(None, [model, clust_u, msm_t, None])

        # Получение распределения на n шагов
        show_distribution_ek(300, model, trajectory, msm, assign_u, clust_u, 0)


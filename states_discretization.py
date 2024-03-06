from deeptime.clustering import KMeans
import deeptime.markov as markov
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import time
import sys

# Кластеризация
def states_clustering(trajectory, method='kmeans_uniform', n_iter_max = 1000, n_cl = 500, dist = 3):
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
    print("Clustering: %s seconds" % (time.time() - start_time))
    return clustering, assignments


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

# Расчет ошибки кластеризации для разного числа кластеров
def mape_of_n_cl(n_clust, trajectory, model, tr_test):
    mape_arr = np.zeros_like(n_clust)
    for i in range(len(n_clust)):
        cur_mape = 0
        for j in range(len(tr_test)):
            clust, assign = states_clustering(trajectory, 'kmeans_uniform', n_iter_max=1000, n_cl=n_clust[i])
            assign_test = clust.transform(tr_test[j])
            cur_mape += calc_mape(model, tr_test[j], clust, assign_test)
        mape_arr[i] = cur_mape/len(tr_test)
    print("MAPEs: ", mape_arr)
    plt.plot(n_clust, mape_arr, 'o--')
    plt.xlabel('$k$')
    plt.ylabel('$MAPE$')
    plt.grid()
    plt.show()

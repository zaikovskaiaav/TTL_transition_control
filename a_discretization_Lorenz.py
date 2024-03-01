from lorenz_model import LorenzModelControl
from a_discretization import get_B, show_B_distribution, get_action_space, show_B
from s_discr_lorenz import show_lorenz_2D
import numpy as np

if __name__ == "__main__":

    m = LorenzModelControl(10, 28, 8 / 3)

    time_step = 0.001
    n_steps = 15000000

    perc_range = 30

    # Нахождение значений правых частей системы B и их распределений для готовой траектории
    trajectory = np.loadtxt('time_series/trajectory_for_clustering_lorenz.txt')

    da = get_B(m, trajectory)

    show_B(da[20:100], scaling=True)
    show_lorenz_2D(trajectory[20:100])

    a_range = show_B_distribution(da, perc_range)
    print(a_range)

    # a_range = np.array([[1, 2, 3],
    #                     [4, 5, 6],
    #                     [7, 8, 9]])

    action_space, actions = get_action_space(a_range, 5, 3)
    print(action_space)
    print(actions)

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
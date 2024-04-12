import contextvars
import numpy as np
import matplotlib.pyplot as plt
import time
# from thequickmath.reduced_models.models import rk4_timestepping
from MFE_model import MoehlisFaisstEckhardtModelControl, rk4_timestepping_control
from s_discretization_MFE import generate_trajectory, random_initial_conditions_mfe
from actions_discretization import get_B, get_B_distribution, get_action_space, show_B


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
    ic = random_initial_conditions_mfe(model.dim, limit=limit)
    trajectory = rk4_timestepping_control(model, ic, action, time_step, n_steps, time_skip=1000, debug=False)
    print("%s seconds" % (time.time() - start_time))
    return trajectory[:-1]

# Генератор траекторий с постоянным управляющим воздействием
def const_control(model, actions):
    for i in range(len(actions)):
        test = generate_trajectory_const(model, time_step, n_steps, actions[i])
        np.savetxt(f'time_series/ad_test{i}.txt', test)


if __name__ == "__main__":
    # Параметры модели
    Re = 500.0
    Lx = 1.75 * np.pi
    Lz = 1.2 * np.pi
    m = MoehlisFaisstEckhardtModelControl(Re, Lx, Lz)

    # Параметры метода Рунге-Кутты
    time_step = 0.001
    n_steps = 15000000

    perc_range = 90 # Величина диапазона возможных действий, в % от распределения

    # Нахождение значений правых частей системы B и их распределений для готовой траектории
    trajectory = np.loadtxt('time_series/trajectory_for_clustering.txt')
    da = get_B(m, trajectory)
    show_B(da)
    a_range = get_B_distribution(da, perc_range, show=True) # Распределение правых частей системы и диапазон возможных действий
    print(a_range)

    action_space, actions = get_action_space(a_range, 5) # Нахождение возможных действий (задается диапазон и число действий для каждой координаты)
    print(action_space)
    print(len(actions))

    #
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
    #     da = get_B(m, trajectory, actions[i])
    #     show_B(da)
    #     show_B_distribution(da)

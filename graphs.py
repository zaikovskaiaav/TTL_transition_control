import numpy as np
import matplotlib.pyplot as plt
from deeptime.clustering import KMeans


''' Графики для MFE '''

# Вывод графиков скоростей
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

# Вывод графиков кинетической энергии
def show_energy(model, trajectory, color=None, ylab="$E(t)$", k=1, line="single"):
    ek = model.kinetic_energy(trajectory)
    # if color is None:
        # color = 'black'
    # color = str(color if color is not None else 'black')
    # print(color)
    if line == "main":
        plt.plot(np.arange(len(ek))*k, ek, linewidth=2.3, color = color if color is not None else 'darkgray',
                 markersize = 0.5, linestyle='solid')
    elif line == "second":
        plt.plot(np.arange(len(ek))*k, ek, linestyle='solid', color = color if color is not None else 'black',
                 linewidth=0.4, markersize = 0.4)
    else:
        plt.plot(np.arange(len(ek)) * k, ek, linewidth=1, color = color if color is not None else 'black',
                 markersize=0.5, linestyle='solid')
    # plt.xlabel("$t$")
    plt.ylabel(ylab)
    # return ek

def show_energy_clust(model, clustering, assignments, color=None, ylab="$E(t)$", k=1, line='single'):
    tr_cl = np.zeros((len(assignments), model.dim))
    for i in range(len(assignments)):
        if assignments[i] == -1:
            tr_cl[i] = tr_cl[i - 1]
        else:
            tr_cl[i] = clustering.cluster_centers[assignments[i]]
    show_energy(model, tr_cl, color, ylab, k, line)

def show_ek(tr, discr):
    plt.figure(figsize=(10, 3))
    if tr and discr:
        show_energy(*tr, line='main')
        show_energy_clust(*discr, line='second')
    elif tr:
        show_energy(*tr, line='single')
    else:
        show_energy_clust(*discr, line='single')
    plt.grid()
    plt.show()



''' Графики для модели Лоренца '''

# Преобразование координат
def coord_lorenz(trajectory):
    x = np.zeros(len(trajectory))
    y = np.zeros(len(trajectory))
    z = np.zeros(len(trajectory))
    for i in range(len(trajectory)):
        x[i] = trajectory[i][0]
        y[i] = trajectory[i][1]
        z[i] = trajectory[i][2]
    # [x, y, z] = np.vsplit(np.transpose(trajectory), 3)
    return x, y, z

# График зависимости координат от времени для модели Лоренца
def plot_data_lorenz_2D(trajectory, axs, labels, line='single', color=None, n_fig=3, delta_t=0.001):
    coords  = coord_lorenz(trajectory)
    col = 'black'
    t = np.arange(len(trajectory))*delta_t
    if line == 'single':
        for i in range(len(coords)):
            axs[i].plot(t, coords[i], color=color if color is not None else 'black', label=labels[i])
    elif line == 'second':
        for i in range(len(coords)):
            axs[i].plot(t, coords[i], '--', color=color if color is not None else 'black', label=labels[i])
    else:
        for i in range(len(coords)):
            axs[i].plot(t, coords[i], color=color if color is not None else 'darkgray', label=labels[i], linewidth=2)
    if line != 'second':
        for i in range(n_fig):
            axs[i].set(ylabel=labels[i])
            axs[i].grid()
    plt.xlabel(r'$t$')

# График x(t) для модели Лоренца
def plot_x_lorenz(trajectory, delta_t=0.001):
    x, y, z  = coord_lorenz(trajectory)
    col = 'black'
    t = np.arange(len(trajectory))*delta_t
    plt.plot(t, x, color=col, label='x(t)')
    plt.ylabel('$x(t)$')
    plt.grid()
    # plt.xlabel(r'$t$')

# График z(x) для модели Лоренца
def plot_xz_lorenz(trajectory, model):
    x, y, z = coord_lorenz(trajectory)
    col = 'black'
    plt.plot(x, z, color=col, label='x(t)')
    plt.plot(np.sqrt(model.beta * (model.Ra-1)),  model.Ra-1, 'x', markersize=8, color='gray')
    plt.plot(-np.sqrt(model.beta * (model.Ra-1)),  model.Ra-1, 'x', markersize=8, color='red')
    plt.ylabel('$z(t)$')
    plt.xlabel('$x(t)$')
    plt.grid()
    # plt.xlabel(r'$t$')

# Графики зависимости координат от времени для непрерывной и дискретной траекторий модели Лоренца
def show_lorenz_xyz_2_lines(trajectory, model, clust, assign):
    n_fig = 3
    fig, axs = plt.subplots(n_fig, figsize=(10, 7))
    labels = ['$x(t)$', '$y(t)$', '$z(t)$']
    tr_cl = np.zeros((len(assign), model.dim))
    for i in range(len(assign)):
        tr_cl[i] = clust.cluster_centers[assign[i]]
    plot_data_lorenz_2D(trajectory, axs, labels, line='main')
    plot_data_lorenz_2D(tr_cl, axs, labels, line='second')
    plt.show()

# Вывод графика траектории системы Лоренца
def show_lorenz_2D(trajectory, model, delta_t=0.001, isx=False, isxz=False):
    n_fig = 3
    labels = ['$x(t)$', '$y(t)$', '$z(t)$']
    # График x(t)
    if isx:
        plt.figure(figsize=(10, 3))
        plot_x_lorenz(trajectory, delta_t)
    # График z(x)
    elif isxz:
        plt.figure(figsize=(7, 4))
        plot_xz_lorenz(trajectory, model)
    # Графики x(t), y(t), z(t)
    else:
        fig, axs = plt.subplots(n_fig, figsize=(10, 6))
        plot_data_lorenz_2D(trajectory, axs, labels, delta_t=delta_t)
    plt.show()

# Вывод графика дискретной траектории системы Лоренца
def show_lorenz_discr_2D(model, clust, assign):
    n_fig = 3
    fig, axs = plt.subplots(n_fig, figsize=(10, 6))
    labels = ['$x(t)$', '$y(t)$', '$z(t)$']
    tr_cl = np.zeros((len(assign), model.dim))
    for i in range(len(assign)):
        tr_cl[i] = clust.cluster_centers[assign[i]]
    plot_data_lorenz_2D(tr_cl, axs, labels)
    plt.show()


# График траектории системы Лоренца 3D
def show_lorenz_3D(trajectory, model, isx=False):
    ax = plt.axes(projection='3d')
    ax.plot3D(np.sqrt(model.beta * (model.Ra-1)), np.sqrt(model.beta * (model.Ra-1)), model.Ra-1, 'x', markersize=4, color='gray')
    ax.plot3D(-np.sqrt(model.beta * (model.Ra-1)), -np.sqrt(model.beta * (model.Ra-1)), model.Ra-1, 'x', markersize=4, color='gray')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')
    x, y, z = coord_lorenz(trajectory)
    ax.plot3D(x, y, z, linewidth=0.7, color='black')
    plt.show()

# График дискретной траектории системы Лоренца 3D
def show_lorenz_3D_discr(model, clust, assign):
    ax = plt.axes(projection ='3d')
    tr_cl = np.zeros((len(assign), model.dim))
    for i in range(len(assign)):
        tr_cl[i] = clust.cluster_centers[assign[i]]
    ax.plot3D(0, 0, 0, 'o', markersize=3, color='red')
    ax.plot3D(np.sqrt(model.beta * (model.Ra-1)), np.sqrt(model.beta * (model.Ra-1)), model.Ra-1, 'o', markersize=3, color='red')
    ax.plot3D(-np.sqrt(model.beta * (model.Ra-1)), -np.sqrt(model.beta * (model.Ra-1)), model.Ra-1, 'o', markersize=3, color='red')

    x, y, z = coord_lorenz(tr_cl)
    ax.plot3D(x, y, z, linewidth=0.7, color='black')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')
    plt.show()

# График кластерных центров 3d для модели Лоренца
def show_lorenz_clusters_3D(trajectory, model, clust, assign):
    ax = plt.axes(projection ='3d')
    tr_cl = np.zeros((len(assign), model.dim))
    for i in range(len(assign)):
        tr_cl[i] = clust.cluster_centers[assign[i]]
    x, y, z = coord_lorenz(trajectory)
    ax.plot3D(x, y, z, 'o', markersize=0.7, color='darkgray')
    x, y, z = coord_lorenz(tr_cl)
    ax.plot3D(x, y, z, 'D', markersize=3, color='black')
    #
    # plot_data_lorenz(trajectory, ax, color='darkgray')
    # plot_data_lorenz(tr_cl, ax, color='black')
    ax.plot3D(0, 0, 0, 'x', markersize=8, color='red')
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
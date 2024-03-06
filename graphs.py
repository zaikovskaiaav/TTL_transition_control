import numpy as np
import matplotlib.pyplot as plt
from deeptime.clustering import KMeans

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
def show_energy(model, trajectory, color=None, ylab="$E(t)$", k=1, linestyle='solid'):
    ek = model.kinetic_energy(trajectory)
    if linestyle == 'solid':
        if color is None:
            color = 'black'
        plt.plot(np.arange(len(ek))*k, ek, linewidth=1.5, color=color, markersize = 0.5)
    else:
        if color is None:
            color = 'gray'
        plt.plot(np.arange(len(ek))*k, ek, linestyle='dotted', color=color, linewidth=1, markersize = 0.5)
    # plt.xlabel("$t$")
    plt.ylabel(ylab)
    # return ek

def show_energy_clust(model, clustering, assignments, color=None, ylab="$E(t)$", k=1, linestyle='solid'):
    tr_cl = np.zeros((len(assignments), model.dim))
    for i in range(len(assignments)):
        if assignments[i] == -1:
            tr_cl[i] = tr_cl[i - 1]
        else:
            tr_cl[i] = clustering.cluster_centers[assignments[i]]
    show_energy(model, tr_cl, color, ylab, k, linestyle)

def show_ek(tr, discr):
    plt.figure(figsize=(10, 3))
    if tr and discr:
        show_energy(*tr, linestyle='solid')
        show_energy_clust(*discr, linestyle='dotted')
    elif tr:
        show_energy(*tr, linestyle='solid')
    else:
        show_energy_clust(*discr, linestyle='solid')
    plt.grid()
    plt.show()
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

# Molecular model parameters
T = 119.8  # [K]   Temperature of system
kB = 1.380649e-23  # [J/K] Boltzmann constant

epsilon = T * kB  # [J]   Lennard-Jones Potential well depth of Argon
sigma = 3.405e-10  # [m]   Lennard-Jones Potential well depth of Argon
# radius_max = 10000  # [-]   *sigma Lennard-Jones Potential cutoff max radius of interaction
Ag_m = 6.6335209e-26  # [kg]  Atomic mass of Argon

# Simulation parameters:
N_particles = 2  # [-]   Amount of particles
h = 0.001  # [-]   Time step
N_time = 1000  # [-]   Number of time steps
box_L = 30  # [-]   *sigma Length of box

potential_energies = np.zeros(N_time)

# grid (0), random distribution (1), two particles (2)
if N_particles == 2:
    test_state = 2
else:
    test_state = 0


def initialize_particles():
    """ This function returns the initial particle state at t=0. The particle
    state contains for each timestep 6 columns (x,y,z,v_x,v_y,v_z) and N rows. """
    particles = np.zeros((N_time, N_particles, 6))
    print("shape of matrix:", particles.shape, "= (t, N, [x y z vx vy vz])")

    if test_state == 0:
        grid_left_boundary = box_L / 2 - 0.8
        grid_right_boundary = box_L / 2 + 1
        box_grid = np.linspace(grid_left_boundary, grid_right_boundary, num=round(N_particles ** (1 / 3)))
        x_grid, y_grid, z_grid = np.meshgrid(box_grid, box_grid, box_grid)
        particles[0, :, 0] = np.matrix.flatten(x_grid)
        particles[0, :, 1] = np.matrix.flatten(y_grid)
        particles[0, :, 2] = np.matrix.flatten(z_grid)
        particles[0, :, 3] = 0  # 1/h
        particles[0, :, 4] = 0  # 1/h
        particles[0, :, 5] = 0  # 1/h
        return particles

    if test_state == 1:
        np.random.seed(0)
        random_left_boundary = box_L / 2 - 1
        random_right_boundary = box_L / 2 + 1
        particles[0, :, 0] = np.random.uniform(random_left_boundary, random_right_boundary, size=N_particles)
        particles[0, :, 1] = np.random.uniform(random_left_boundary, random_right_boundary, size=N_particles)
        particles[0, :, 2] = np.random.uniform(random_left_boundary, random_right_boundary, size=N_particles)
        return particles

    if test_state == 2:
        v_init_two_particles = 10
        particles[0, 0, 0], particles[0, 0, 1] = [box_L / 2 + 3, box_L / 2 + 1]  # + box_L / 600]
        particles[0, 1, 0], particles[0, 1, 1] = [box_L / 2 - 3, box_L / 2 - 1]  # - box_L / 600]
        particles[0, 0, 3], particles[0, 0, 4] = [-v_init_two_particles, 0]
        particles[0, 1, 3], particles[0, 1, 4] = [v_init_two_particles, 0]
        return particles


def lj_force(r):
    """ Given the unitless radius between two particles,
    this returns the unitless Lennard Jones Force. """
    force = -4 * (-12 * (r ** -14) + 6 * (r ** -8))
    return force


def potential_energy(r):
    potential = 4 * (r ** -12 - r ** -6)
    return potential


def mic(x1, x2, y1, y2, z1, z2):
    """ Given the x,y,z coordinates of two particles, the minimum image
    convention is applied, returning the closest x,y,z coordinates."""
    # x_close, y_close, z_close = x2, y2, z2
    # if x2 - x1 > box_L / 2:
    #     x_close = x2 - box_L
    # elif x2 - x1 <= -box_L / 2:
    #     x_close = x2 + box_L
    # if y2 - y1 > box_L / 2:
    #     y_close = y2 - box_L
    # elif y2 - y1 <= -box_L / 2:
    #     y_close = y2 + box_L
    # if z2 - z1 > box_L / 2:
    #     z_close = z2 - box_L
    # elif z2 - z1 <= -box_L / 2:
    #     z_close = z2 + box_L
    x_close = x1 + ((x2 - x1 + box_L / 2) % box_L - box_L / 2)
    y_close = y1 + ((y2 - y1 + box_L / 2) % box_L - box_L / 2)
    z_close = z1 + ((z2 - z1 + box_L / 2) % box_L - box_L / 2)
    return x_close, y_close, z_close


def abs_distance(x1, x2, y1, y2, z1, z2):
    """ Given the x,y,z coordinates of two particles, this returns the
    absolute distance to the closest partner on the infinite canvas."""
    x2, y2, z2 = mic(x1, x2, y1, y2, z1, z2)
    r_abs = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    return r_abs


def unit_direction(r_abs, x1, x2, y1, y2, z1, z2):
    """ Given the absolute relative distance and x,y,z coordinates
    of two particles, this returns the unit directional vector. """
    r_x = (x1 - x2) / r_abs
    r_y = (y1 - y2) / r_abs
    r_z = (z1 - z2) / r_abs
    return r_x, r_y, r_z


def periodicity_warp(x_new, y_new, z_new):
    """ Given the updated x,y,z coordinates of a particle, this applies
    periodic boundary conditions, returning the new warped coordinates."""
    if x_new >= box_L:
        x_new = x_new % box_L
    if x_new < 0:
        x_new = box_L - (abs(x_new) % box_L)
    if y_new >= box_L:
        y_new = y_new % box_L
    if y_new < 0:
        y_new = box_L - (abs(y_new) % box_L)
    if z_new >= box_L:
        z_new = z_new % box_L
    if z_new < 0:
        z_new = box_L - (abs(z_new) % box_L)
    return x_new, y_new, z_new


def interaction_force(i, j, particles):
    """ Given the time index i, particle index j, and particle state,
        this returns the old x,y,z Force components."""
    x_old, y_old, z_old, vx_old, vy_old, vz_old = particles[i - 1][j]
    f_x, f_y, f_z = [0, 0, 0]
    for k in range(N_particles):
        radius = abs_distance(x_old, particles[i - 1][k][0], y_old, particles[i - 1][k][1], z_old,
                              particles[i - 1][k][2])
        if (k != j) and (radius != 0):
            force = lj_force(radius)
            r_unit = unit_direction(radius, x_old, particles[i - 1][k][0], y_old, particles[i - 1][k][1], z_old,
                                    particles[i - 1][k][2])
            f_x += force * r_unit[0]
            f_y += force * r_unit[1]
            f_z += force * r_unit[2]
    return f_x, f_y, f_z


def update_position(i, j, particles, f_x_old, f_y_old, f_z_old):
    """ Given the time index i, particle index j, and particle state,
        this returns the new x,y,z coordinates based on Newtonian dynamics and the Verlet algorithm."""
    x_old, y_old, z_old, vx_old, vy_old, vz_old = particles[i - 1][j]
    # x_new = x_old + vx_old * h + (h ** 2 / (2 * Ag_m)) * f_x_old
    x_new = x_old + vx_old * h + (h ** 2 / 2) * f_x_old
    y_new = y_old + vy_old * h + (h ** 2 / 2) * f_y_old
    z_new = z_old + vz_old * h + (h ** 2 / 2) * f_z_old
    x_new, y_new, z_new = periodicity_warp(x_new, y_new, z_new)
    return x_new, y_new, z_new


def update_velocity(i, j, particles, f_x_old, f_y_old, f_z_old):
    """ Given the time index i, particle index j, and particle state,
    this returns the new x,y,z velocities based on Newtonian dynamics and the Verlet algorithm."""
    x_old, y_old, z_old, vx_old, vy_old, vz_old = particles[i - 1][j]
    x_new, y_new, z_new, vx_new, vy_new, vz_new = particles[i][j]
    f_x, f_y, f_z = [0, 0, 0]

    for k in range(N_particles):
        radius = abs_distance(x_new, particles[i][k][0], y_new, particles[i][k][1], z_new, particles[i][k][2])
        if (k != j) and (radius != 0):
            potential_energies[i] += potential_energy(radius)
            force = lj_force(radius)
            r_unit = unit_direction(radius, x_new, particles[i][k][0], y_new, particles[i][k][1], z_new,
                                    particles[i][k][2])
            f_x += force * r_unit[0]
            f_y += force * r_unit[1]
            f_z += force * r_unit[2]

    vx_new = vx_old + (h / 2) * (f_x_old + f_x)
    vy_new = vy_old + (h / 2) * (f_y_old + f_y)
    vz_new = vz_old + (h / 2) * (f_z_old + f_z)
    return vx_new, vy_new, vz_new


def update_particles(i, particles):
    """ This function updates all particles. """
    for j in range(N_particles):
        for k in range(N_particles):
            f_x, f_y, f_z = interaction_force(i, k, particles)
            x_new, y_new, z_new = update_position(i, k, particles, f_x, f_y, f_z)
            particles[i][k][0:3] = [x_new, y_new, z_new]
        f_x, f_y, f_z = interaction_force(i, j, particles)
        vx_new, vy_new, vz_new = update_velocity(i, j, particles, f_x, f_y, f_z)
        particles[i][j][3:6] = [vx_new, vy_new, vz_new]

    return particles


def kinetic_energy(velocities):
    """ Given the x,y,z velocities, this returns the dimensioneless kinetic energy. """
    v_x, v_y, v_z = velocities
    v_abs = np.sqrt(v_x ** 2 + v_y ** 2 + v_z ** 2)
    return 0.5 * v_abs ** 2


def plot_kinetic_energy(particles):
    """ Given the particle state, this plots the kinetic energy. """
    total_kinetic_energies = np.zeros(N_time)
    for i in range(N_time):
        energy_kin = 0
        for j in range(N_particles):
            energy_kin += kinetic_energy(particles[i, j, 3:])
        total_kinetic_energies[i] = energy_kin

    plt.figure(figsize=(5, 4))
    plt.title("Total Kinetic Energy")
    plt.plot(np.arange(0, N_time), total_kinetic_energies)
    plt.xlabel("time point i")
    plt.savefig("kinetic_energies.png")
    plt.close()
    return total_kinetic_energies


def plot_potential_energy():
    plt.figure(figsize=(5, 4))
    plt.title("Total Potential Energy")
    plt.plot(np.arange(0, N_time), potential_energies)
    plt.xlabel("time")
    plt.ylabel("potential energy")
    plt.savefig("potential_energies.png")
    plt.close()
    return None


def plot_total_energy(particles):
    plt.figure(figsize=(5, 4))
    plt.title("Total Potential Energy")
    kin_en = plot_kinetic_energy(particles)
    plt.plot(np.arange(0, N_time), potential_energies, label="potential")
    plt.plot(np.arange(0, N_time), kin_en - np.average(kin_en),
             label="kinetic - $E_{kin, avg}$")
    # plt.plot(np.arange(0, N_time), potential_energies+total_kinetic_energy_list, label="total")
    plt.xlabel("time")
    plt.ylabel("energy")
    plt.legend()
    plt.savefig("total_energies.png")
    plt.close()
    return None


def plot_z_slice(particles):
    plt.figure()
    plt.title("z = 0 projection")
    plt.plot(particles[:, 0, 0], particles[:, 0, 1], "o-", markersize=1, c="b")
    plt.plot(particles[:, 1, 0], particles[:, 1, 1], "o-", markersize=1, c="r")
    plt.plot(particles[0, 0, 0], particles[0, 0, 1], "s", markersize=10, c="b")
    plt.plot(particles[0, 1, 0], particles[0, 1, 1], "s", markersize=10, c="r")
    plt.xlim(0, box_L)
    plt.ylim(0, box_L)
    plt.show()


def generate_gif(particles):
    fig = plt.figure(figsize=(3, 3), dpi=100)
    ax = plt.axes(xlim=(0, box_L), ylim=(0, box_L), zlim=(0, box_L), projection='3d')
    ax.view_init(20, 30)
    scatter = ax.scatter([], [], [], s=5, marker='o')
    iteration = ax.text(box_L, 0, box_L, "i=0", color="red")

    def update(i):
        scatter._offsets3d = (particles[i, :, 0], particles[i, :, 1], particles[i, :, 2])
        iteration.set_text("i=" + str(i))
        return scatter, iteration,

    anim = animation.FuncAnimation(fig, update, frames=N_time, interval=1)
    anim.save('MD_simulation.gif', fps=80)
    plt.show()
    # plt.close()


def main():
    """ Main function to be run for this code. """
    particles = initialize_particles()
    for i in range(1, N_time):
        print("\rProgress: {}%".format(round((100 * (i + 1) / N_time))), end='')
        particles = update_particles(i, particles)
    return particles


particles_simulation = main()

plot_potential_energy()
plot_kinetic_energy(particles_simulation)
plot_total_energy(particles_simulation)
if N_particles == 2:
    plot_z_slice(particles_simulation)
else:
    generate_gif(particles_simulation)

# generate_gif(particles_simulation)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

# Molecular model parameters
T = 119.8  # [K]   Temperature of system
kB = 1.380649e-23  # [J/K] Boltzmann constant

epsilon = T * kB  # [J]   Lennard-Jones Potential well depth of Argon
sigma = 3.405e-10  # [m]   Lennard-Jones Potential well depth of Argon
radius_max = 3  # [-]   *sigma Lennard-Jones Potential cutoff max radius of interaction
Ag_m = 6.6335209e-26  # [kg]  Atomic mass of Argon

# Simulation parameters:
N_particles = 3 ** 3  # [-]   Amount of particles
h = 0.01  # [-]   Time step
N_time = 100  # [-]   Number of time steps
box_L = 30  # [-]   *sigma Length of box
# grid (0), random distribution (1), two particles (2)
test_state = 0


def lj_force(r):
    """ Given the unitless radius between two particles,
    this returns the unitless Lennard Jones Force. """
    force = -4 * (-12 * (r ** -14) + 6 * (r ** -8))
    return force


def direction(r_abs, x1, x2, y1, y2, z1, z2):
    """ Given the absolute relative distance and x,y,z coordinates
    of two particles, this returns the unit directional vector. """
    r_x = (x2 - x1) / r_abs
    r_y = (y2 - y1) / r_abs
    r_z = (z2 - z1) / r_abs
    return r_x, r_y, r_z


def mic(x1, x2, y1, y2, z1, z2):
    """ Given the x,y,z coordinates of two particles, the minimum image
    convention is applied, returning the closest x,y,z coordinates."""
    x_close, y_close, z_close = x2, y2, z2

    # test_x_close = (x2 - x1 + box_L / 2) % box_L - box_L / 2
    # test_y_close = (y2 - y1 + box_L / 2) % box_L - box_L / 2
    # test_z_close = (z2 - z1 + box_L / 2) % box_L - box_L / 2

    if x2 - x1 > box_L / 2:
        x_close = x2 - box_L
    elif x2 - x1 <= -box_L / 2:
        x_close = x2 + box_L
    if y2 - y1 > box_L / 2:
        y_close = y2 - box_L
    elif y2 - y1 <= -box_L / 2:
        y_close = y2 + box_L
    if z2 - z1 > box_L / 2:
        z_close = z2 - box_L
    elif z2 - z1 <= -box_L / 2:
        z_close = z2 + box_L
    return x_close, y_close, z_close


def distance(x1, x2, y1, y2, z1, z2):
    """ Given the x,y,z coordinates of two particles, this returns the
    absolute distance to the closest partner on the infinite canvas."""
    x2, y2, z2 = mic(x1, x2, y1, y2, z1, z2)
    r_abs = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    return r_abs


def periodicity(x_new, y_new, z_new):
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


def initialize_particles():
    """ This function returns the initial particle state at t=0. The particle
    state contains for each timestep 6 columns (x,y,z,v_x,v_y,v_z) and N rows. """
    particles = np.zeros((N_time, N_particles, 6))
    print("shape of matrix:", particles.shape, "= (t, N, [x y z vx vy vz])")

    if test_state == 0:
        grid_left_boundary = box_L / 2 - 0.6
        grid_right_boundary = box_L / 2 + 0.6
        box_grid = np.linspace(grid_left_boundary, grid_right_boundary, num=round(N_particles ** (1 / 3)))
        x_grid, y_grid, z_grid = np.meshgrid(box_grid, box_grid, box_grid)
        particles[0, :, 0] = np.matrix.flatten(x_grid)
        particles[0, :, 1] = np.matrix.flatten(y_grid)
        particles[0, :, 2] = np.matrix.flatten(z_grid)
        return particles

    if test_state == 1:
        np.random.seed(0)
        random_left_boundary = (2 / 5) * box_L
        random_right_boundary = (3 / 5) * box_L
        particles[0, :, 0] = np.random.uniform(random_left_boundary, random_right_boundary, size=N_particles)
        particles[0, :, 1] = np.random.uniform(random_left_boundary, random_right_boundary, size=N_particles)
        particles[0, :, 2] = np.random.uniform(random_left_boundary, random_right_boundary, size=N_particles)
        # random_sigma = 1
        # random_mu = box_L/2
        # particles[0, :, 0] = np.random.normal(loc=random_mu, scale=random_sigma, size=N_particles)
        # particles[0, :, 1] = np.random.normal(loc=random_mu, scale=random_sigma, size=N_particles)
        # particles[0, :, 2] = np.random.normal(loc=random_mu, scale=random_sigma, size=N_particles)
        return particles

    if test_state == 2:
        particles[0, 0, 0], particles[0, 0, 1] = [box_L / 2 + 2, box_L / 2]
        particles[0, 1, 0], particles[0, 1, 1] = [box_L / 2 - 1, box_L / 2]
        particles[0, 0, 3], particles[0, 0, 4] = [0, 0]
        particles[0, 1, 3], particles[0, 1, 4] = [0, 0]
        return particles


def update_velocity(i, j, particles):
    """ Given the time index i, particle index j, and particle state,
    this returns the new x,y,z velocities based on Newtonian dynamics."""
    x_old, y_old, z_old, vx_old, vy_old, vz_old = particles[i - 1][j]
    vx_new = vx_old
    vy_new = vy_old
    vz_new = vz_old
    for k in range(N_particles):
        radius = distance(x_old, particles[i - 1][k][0], y_old, particles[i - 1][k][1], z_old, particles[i - 1][k][2])
        if (k != j) and (radius <= radius_max) and (radius != 0):
            force = lj_force(radius)
            radius_direction = direction(radius, x_old, particles[i - 1][k][0], y_old, particles[i - 1][k][1], z_old,
                                         particles[i - 1][k][2])
            vx_new = vx_old + force * h * radius_direction[0]
            vy_new = vy_old + force * h * radius_direction[1]
            vz_new = vz_old + force * h * radius_direction[2]
    return vx_new, vy_new, vz_new


def update_particle(i, j, particles):
    """ Given the time index i, particle index j, and particle state,
    this returns the new x,y,z coordinates based on Newtonian dynamics."""
    x_old, y_old, z_old, vx_old, vy_old, vz_old = particles[i - 1][j]
    x_new = x_old + vx_old * h
    y_new = y_old + vy_old * h
    z_new = z_old + vz_old * h

    x_new, y_new, z_new = periodicity(x_new, y_new, z_new)
    vx_new, vy_new, vz_new = update_velocity(i, j, particles)
    if x_new >= box_L or y_new >= box_L or x_new < 0 or y_new < 0 or z_new < 0 or z_new >= box_L:
        # if next position is outside box
        particles[i][j] = x_new, y_new, z_new, vx_old, vy_old, vz_old
        return particles
    particles[i][j] = x_new, y_new, z_new, vx_new, vy_new, vz_new

    return particles


def update_particles(i, particles):
    """ This function updates all particles. """
    for j in range(N_particles):
        update_particle(i, j, particles)
    return particles


def kinetic_energy(velocities):
    """ Given the x,y,z velocities, this returns the dimensioneless kinetic energy. """
    v_x, v_y, v_z = velocities
    v_abs = np.sqrt(v_x ** 2 + v_y ** 2 + v_z ** 2)
    return 0.5 * Ag_m * v_abs ** 2 / epsilon


def plot_kinetic_energy(particles):
    """ Given the particle state, this plots the kinetic energy. """
    total_kinetic_energy_list = []
    for particles_t in particles:
        total_kinetic_energy = 0
        for particle_t in particles_t:
            # print(particle_t)
            total_kinetic_energy += kinetic_energy(particle_t[3:])
        total_kinetic_energy_list.append(total_kinetic_energy)

    plt.figure(figsize=(5, 4))
    plt.title("Total Kinetic Energy")
    plt.plot(np.arange(0, N_time), total_kinetic_energy_list)
    # plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("time point i")
    plt.savefig("kinetic_energies.png")
    plt.close()
    return None


def main():
    """ Main function to be run for this code. """
    particles = initialize_particles()
    for i in range(1, N_time):
        print("\rProgress: {}%".format(round((100 * (i + 1) / N_time))), end='')
        particles = update_particles(i, particles)
    return particles


fig = plt.figure(figsize=(3, 3), dpi=100)
ax = plt.axes(xlim=(0, box_L), ylim=(0, box_L), zlim=(0, box_L), projection='3d')
ax.view_init(20, 30)
scatter = ax.scatter([], [], [], s=5)
iteration = ax.text(box_L, 0, box_L, "i=0", color="red")
particles_simulation = main()


def update(i):
    scatter._offsets3d = (particles_simulation[i, :, 0], particles_simulation[i, :, 1], particles_simulation[i, :, 2])
    iteration.set_text("i=" + str(i))
    return scatter, iteration,


anim = animation.FuncAnimation(fig, update, frames=N_time, interval=10)  # , interval=500)
anim.save('MD_simulation.gif', writer='Pillow', fps=30)
plt.show()
plot_kinetic_energy(particles_simulation)

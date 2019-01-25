import sys
sys.path.append('./negf')
sys.path.append('../negf')
import matplotlib.pyplot as plt
import numpy as np
import tb
from field import Field


def compute_tb_matrices(save=True):
    tb.Atom.orbital_sets = {'Si': 'SiliconSP3D5S', 'H': 'HydrogenS'}
    h = tb.Hamiltonian(xyz='/home/mk/TB_project/input_samples/SiNW2.xyz', nn_distance=2.4)
    h.initialize()
    period = [0, 0, 5.50]
    h.set_periodic_bc([period])
    h_l, h_0, h_r = h.get_coupling_hamiltonians()

    coords = h.get_site_coordinates()

    if save:
        np.save('h_l', h_l)
        np.save('h_0', h_0)
        np.save('h_r', h_r)
        np.save('coords', coords)

    return h_l, h_0, h_r, coords


def compute_self_energies_for_leads(energy, h_l, h_0, h_r, save=True):

    sgf_l = []
    sgf_r = []
    num_sites = h_0.shape[0]

    for j, E in enumerate(energy):
        L, R = tb.surface_greens_function(E, h_l, h_0, h_r)

        test_gf = E * np.identity(num_sites) - h_0 - L - R
        metrics = np.linalg.cond(test_gf)
        print("{} of {}: energy is {}, metrics is {}".format(j + 1, energy.shape[0], E, metrics))

        # if metrics > 15000:
        #     R = iterate_gf(E, h_0, h_l, h_r, R, 1)
        #     L = iterate_gf(E, h_0, h_l, h_r, L, 1)

        sgf_l.append(L)
        sgf_r.append(R)
        # factor.append(phase)
        # factor1.append(phase1)
        # factor2.append(phase2)

    sgf_l = np.array(sgf_l)
    sgf_r = np.array(sgf_r)

    if save:
        np.save('sgf_l', sgf_l)
        np.save('sgf_r', sgf_r)

    return sgf_l, sgf_r


# h_l, h_0, h_r, coords = compute_tb_matrices()

h_l = np.load('h_l.npy')
h_0 = np.load('h_0.npy')
h_r = np.load('h_r.npy')
coords = np.load('coords.npy')

energy = np.linspace(2.1, 2.2, 50)

# sgf_l, sgf_r = compute_self_energies_for_leads(energy, h_l, h_0, h_r, save=True)

sgf_l = np.load('sgf_l.npy')
sgf_r = np.load('sgf_r.npy')

num_sites = h_0.shape[0]

field = Field(path='/home/mk/gaussian_swarm/gauss_comp/out_ionized.cube')
# field.set_origin(np.array([6.36, 11.86 + 5, 2.75]))

period = np.array([0, 0, 5.50])
values_m4 = field.get_values(coords, translate=-4 * period)
values_m3 = field.get_values(coords, translate=-3 * period)
values_m2 = field.get_values(coords, translate=-2 * period)
values_m1 = field.get_values(coords, translate=-period)
values_0 = field.get_values(coords)
values_1 = field.get_values(coords, translate=period)
values_2 = field.get_values(coords, translate=2 * period)
values_3 = field.get_values(coords, translate=3 * period)
values_4 = field.get_values(coords, translate=4 * period)

eps = 12.0

shifts = np.concatenate(
    (values_4, values_3, values_2, values_1, values_0, values_m1, values_m2, values_m3, values_m4)) / eps

coords1 = np.concatenate((coords - 4 * period,
                          coords - 3 * period,
                          coords - 2 * period,
                          coords - period,
                          coords,
                          coords + period,
                          coords + 2 * period,
                          coords + 3 * period,
                          coords + 4 * period))

from mpl_toolkits.mplot3d import axes3d

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d', proj_type='ortho')
ax.set_xlim3d(0, 60)
ax.set_ylim3d(0, 60)
ax.set_zlim3d(-30, 30)
ax.view_init(0, 90)
ax.scatter(coords1[:, 0], coords1[:, 1], coords1[:, 2], c=shifts, s=100)

zeros = np.zeros(h_0.shape)

# h_d = np.block([[h_0 + np.diag(values_m2), h_l, zeros, zeros, zeros],
#                 [h_r, h_0 + np.diag(values_m1), h_l, zeros, zeros],
#                 [zeros, h_r, h_0 + np.diag(values_0), h_l, zeros],
#                 [zeros, zeros, h_r, h_0 + np.diag(values_1), h_l],
#                 [zeros, zeros, zeros, h_r, h_0 + np.diag(values_2)]])

h_d = np.block([[h_0, h_l, zeros, zeros, zeros],
                [h_r, h_0, h_l, zeros, zeros],
                [zeros, h_r, h_0, h_l, zeros],
                [zeros, zeros, h_r, h_0, h_l],
                [zeros, zeros, zeros, h_r, h_0]])

# h_d -= np.diag(shifts)

damp = np.diag(np.zeros(np.diag(h_d).shape) + 1j * 0.001)

# gf = np.linalg.pinv(np.multiply.outer(energy, np.identity(2 * num_sites)) - h_device - selfenergy)

# gf = np.linalg.pinv(np.multiply.outer(energy, np.identity(2*num_sites)) - h_d - sgf_l - sgf_r)

# gf = regularize_gf(gf)

# dos = -np.trace(np.imag(gf), axis1=1, axis2=2)

energy = energy[5:25]

tr = np.zeros((energy.shape[0]), dtype=np.complex)
dos = np.zeros((energy.shape[0]), dtype=np.complex)

from negf.recursive_greens_functions import recursive_gf

mat_d_list = [h_0, h_0, h_0, h_0]
mat_u_list = [h_l, h_l, h_l]
mat_l_list = [h_r, h_r, h_r]

for j, E in enumerate(energy):
    print(j)

    mat_d_list[0] = mat_d_list[0] + sgf_l[j, :, :]
    mat_d_list[-1] = mat_d_list[-1] + sgf_r[j, :, :]

    grd, grl, gru, gr_left = recursive_gf(E, mat_d_list, mat_u_list, mat_l_list)

    mat_d_list[0] = mat_d_list[0] - sgf_l[j, :, :]
    mat_d_list[-1] = mat_d_list[-1] - sgf_r[j, :, :]

    for jj in range(len(grd)):
        dos[j] += np.real(np.trace(1j * (grd[jj] - grd[jj].H)))

        # gamma_l = 1j * (np.matrix(sgf_l_loc) - np.matrix(sgf_l_loc).H)
        # gamma_r = 1j * (np.matrix(sgf_r_loc) - np.matrix(sgf_r_loc).H)
        # dos[j] = np.real(np.trace(1j * (gf0 - gf0.H)))
        # tr[j] = np.real(np.trace(gamma_l * gf0 * gamma_r * gf0.H))


# for j, E in enumerate(energy):
#     sgf_l_loc = np.block([[sgf_l[j, :, :], zeros, zeros, zeros, zeros],
#                           [zeros, zeros, zeros, zeros, zeros],
#                           [zeros, zeros, zeros, zeros, zeros],
#                           [zeros, zeros, zeros, zeros, zeros],
#                           [zeros, zeros, zeros, zeros, zeros]])
#
#     sgf_r_loc = np.block([[zeros, zeros, zeros, zeros, zeros],
#                           [zeros, zeros, zeros, zeros, zeros],
#                           [zeros, zeros, zeros, zeros, zeros],
#                           [zeros, zeros, zeros, zeros, zeros],
#                           [zeros, zeros, zeros, zeros, sgf_r[j, :, :]]])
#
#     gf = E * np.identity(5 * num_sites) - h_d - sgf_l_loc - sgf_r_loc
#
#     metrics = np.linalg.cond(gf)
#     print("{} of {}: energy is {}, metrics is {}".format(j + 1, energy.shape[0], E, metrics))
#
#     gf = np.linalg.pinv(gf)
#
#     gf0 = np.matrix(gf)
#     gamma_l = 1j * (np.matrix(sgf_l_loc) - np.matrix(sgf_l_loc).H)
#     gamma_r = 1j * (np.matrix(sgf_r_loc) - np.matrix(sgf_r_loc).H)
#     dos[j] = np.real(np.trace(1j * (gf0 - gf0.H)))
#     tr[j] = np.real(np.trace(gamma_l * gf0 * gamma_r * gf0.H))

ax = plt.axes()
ax.set_xlabel('Energy (eV)')
ax.set_ylabel('DOS')
ax.plot(energy, dos)
plt.show()

ax = plt.axes()
ax.set_xlabel('Energy (eV)')
ax.set_ylabel('Transmission coefficient (a.u.)')
ax.plot(energy, tr)
plt.show()

plt.plot(dos)
plt.show()

print(sgf_l.shape)

energy1 = np.pad(energy, (307, 307), 'symmetric', reflect_type='odd')
energy1 = np.unique(energy1)
tr1 = np.pad(tr, (300, 300), 'edge')
kb = 8.6e-5
T = 300

fd = 1.0 / (1.0 + np.exp((energy1 - energy[10]) / kb / T))
fd_div = -np.diff(fd) / np.diff(energy1)
cond = np.convolve(tr1, fd_div, mode='same')

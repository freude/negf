import sys
sys.path.append('./negf')
sys.path.append('../negf')
import matplotlib.pyplot as plt
import numpy as np
import tb
from field import Field


def compute_tb_matrices(save=True):

    tb.Atom.orbital_sets = {'Si': 'SiliconSP3D5S', 'H': 'HydrogenS'}
    h = tb.Hamiltonian(xyz='/home/mk/NEGF_project/SiNW.xyz', nn_distance=2.4)
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


# h_l, h_0, h_r, coords = compute_tb_matrices()

h_l = np.load('h_l.npy')
h_0 = np.load('h_0.npy')
h_r = np.load('h_r.npy')
coords = np.load('coords.npy')

energy = np.linspace(2.1, 2.2, 50)


def compute_self_energies_for_leads(energy, h_l, h_0, h_r, save=True):

    sgf_l = []
    sgf_r = []
    factor = []
    factor1 = []
    factor2 = []

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


# sgf_l, sgf_r = compute_self_energies_for_leads(energy, h_l, h_0, h_r, save=True)

sgf_l = np.load('sgf_l.npy')
sgf_r = np.load('sgf_r.npy')


num_sites = h_0.shape[0]

field = Field(path='/home/mk/gaussian_swarm/gauss_comp/out_ionized.cube')
field.set_origin(np.array([6.36, 11.86+10, 2.75]))

period = np.array([0, 0, 5.50])
values_m2 = field.get_values(coords, translate=-2*period)
values_m1 = field.get_values(coords, translate=-period)
values_0 = field.get_values(coords)
values_1 = field.get_values(coords, translate=period)
values_2 = field.get_values(coords, translate=2*period)

eps = 7.0

shifts = np.concatenate((values_m2, values_m1, values_0, values_1, values_2)) / eps

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

h_d += np.diag(shifts)

damp = np.diag(np.zeros(np.diag(h_d).shape) + 1j*0.001)

# gf = np.linalg.pinv(np.multiply.outer(energy, np.identity(2 * num_sites)) - h_device - selfenergy)

# gf = np.linalg.pinv(np.multiply.outer(energy, np.identity(2*num_sites)) - h_d - sgf_l - sgf_r)

# gf = regularize_gf(gf)

# dos = -np.trace(np.imag(gf), axis1=1, axis2=2)

tr = np.zeros((energy.shape[0]), dtype=np.complex)
dos = np.zeros((energy.shape[0]), dtype=np.complex)

plt.axis([0, np.max(energy), 0, 3])

for j, E in enumerate(energy[:25]):

    sgf_l_loc = np.block([[sgf_l[j, :, :], zeros, zeros, zeros, zeros],
                          [zeros, zeros, zeros, zeros, zeros],
                          [zeros, zeros, zeros, zeros, zeros],
                          [zeros, zeros, zeros, zeros, zeros],
                          [zeros, zeros, zeros, zeros, zeros]])

    sgf_r_loc = np.block([[zeros, zeros, zeros, zeros, zeros],
                          [zeros, zeros, zeros, zeros, zeros],
                          [zeros, zeros, zeros, zeros, zeros],
                          [zeros, zeros, zeros, zeros, zeros],
                          [zeros, zeros, zeros, zeros, sgf_r[j, :, :]]])

    gf = E * np.identity(5 * num_sites) - h_d - sgf_l_loc - sgf_r_loc

    metrics = np.linalg.cond(gf)
    print("{} of {}: energy is {}, metrics is {}".format(j + 1, energy.shape[0], E, metrics))

    gf = np.linalg.pinv(gf)

    gf0 = np.matrix(gf)
    gamma_l = 1j * (np.matrix(sgf_l_loc) - np.matrix(sgf_l_loc).H)
    gamma_r = 1j * (np.matrix(sgf_r_loc) - np.matrix(sgf_r_loc).H)
    dos[j] = np.real(np.trace(1j * (gf0 - gf0.H)))
    tr[j] = np.real(np.trace(gamma_l * gf0 * gamma_r * gf0.H))

    plt.scatter(E, tr[j])
    plt.pause(0.05)


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
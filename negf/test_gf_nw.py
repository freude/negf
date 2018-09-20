import matplotlib.pyplot as plt
import numpy as np
import tb
from field import Field


tb.Atom.orbital_sets = {'Si': 'SiliconSP3D5S', 'H': 'HydrogenS'}
h = tb.Hamiltonian(xyz='/home/mk/NEGF_project/SiNW.xyz', nn_distance=2.4)
h.initialize()
period = [0, 0, 5.50]
h.set_periodic_bc([period])
h_l, h_0, h_r = h.get_coupling_hamiltonians()

# energy = np.linspace(2.07, 2.3, 50)
# energy = np.linspace(2.07, 2.3, 50) + 0.2
# energy = np.linspace(-1.0, -0.3, 200)
energy = np.concatenate((np.linspace(-0.8, -0.3, 100), np.linspace(2.05, 2.5, 100)))
energy = np.linspace(2.05, 2.5, 50)
# energy = np.linspace(2.3950, 2.4050, 20)

# energy = energy[20:35]

sgf_l = []
sgf_r = []
factor = []
factor1 = []
factor2 = []

num_sites = h_0.shape[0]

for j, E in enumerate(energy):
    # L, R = surface_greens_function_poles_Shur(j, E, h_l, h_0, h_r)
    L, R = tb.surface_greens_function(E, h_l, h_0, h_r)

    test_gf = E * np.identity(num_sites) - h_0 - L - R

    metrics = np.linalg.cond(test_gf)
    print "{} of {}: energy is {}, metrics is {}".format(j + 1, energy.shape[0], E, metrics)

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
factor = np.array(factor)
factor1 = np.array(factor1)
factor2 = np.array(factor2)

# field = Field(path='/home/mk/gaussian_swarm/gauss_comp/out_neutral.cube')
# coords = h.get_site_coordinates()
# period = np.array(period)
# values_m2 = field.get_values(coords, translate=-2*period)
# values_m1 = field.get_values(coords, translate=-period)
# values_0 = field.get_values(coords)
# values_1 = field.get_values(coords, translate=period)
# values_2 = field.get_values(coords, translate=2*period)
#
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

# gf = np.linalg.pinv(np.multiply.outer(energy, np.identity(2 * num_sites)) - h_device - selfenergy)

# gf = np.linalg.pinv(np.multiply.outer(energy, np.identity(2*num_sites)) - h_d - sgf_l - sgf_r)

# gf = regularize_gf(gf)

# dos = -np.trace(np.imag(gf), axis1=1, axis2=2)

tr = np.zeros((energy.shape[0]), dtype=np.complex)
dos = np.zeros((energy.shape[0]), dtype=np.complex)

for j, E in enumerate(energy):
    print(j)
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

    gf = np.linalg.pinv(E * np.identity(5 * num_sites) - h_d - sgf_l_loc - sgf_r_loc)

    gf0 = np.matrix(gf)
    gamma_l = 1j * (np.matrix(sgf_l_loc) - np.matrix(sgf_l_loc).H)
    gamma_r = 1j * (np.matrix(sgf_r_loc) - np.matrix(sgf_r_loc).H)
    dos[j] = np.real(np.trace(1j * (gf0 - gf0.H)))
    tr[j] = np.real(np.trace(gamma_l * gf0 * gamma_r * gf0.H))

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

print sgf_l.shape
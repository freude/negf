import numpy as np
import tb
from negf.hamiltonian_chain import HamiltonianChain
from negf.field import Field
from negf.recursive_greens_functions import recursive_gf


def compute_self_energies_for_leads(energy, h_l, h_0, h_r, save=True):

    sgf_l = []
    sgf_r = []
    num_sites = h_0.shape[0]

    for j, E in enumerate(energy):

        L, R = tb.surface_greens_function(E, h_l, h_0, h_r, iterate=True)

        test_gf = E * np.identity(num_sites) - h_0 - L - R
        metrics = np.linalg.cond(test_gf)
        print("{} of {}: energy is {}, metrics is {}".format(j + 1, energy.shape[0], E, metrics))

        sgf_l.append(L)
        sgf_r.append(R)


    sgf_l = np.array(sgf_l)
    sgf_r = np.array(sgf_r)

    if save:
        np.save('sgf_l', sgf_l)
        np.save('sgf_r', sgf_r)

    return sgf_l, sgf_r


h_l = np.load('h_l.npy')
h_0 = np.load('h_0.npy')
h_r = np.load('h_r.npy')
coords = np.load('coords.npy')

energy = np.linspace(2.1, 2.2, 50)

# sgf_l, sgf_r = compute_self_energies_for_leads(energy, h_l, h_0, h_r, save=True)
sgf_l = np.load('sgf_l.npy')
sgf_r = np.load('sgf_r.npy')

num_periods = 3

h_chain = HamiltonianChain(h_l, h_0, h_r, coords)
h_chain.translate([[0, 0, 5.50]], num_periods, num_periods)

field = Field(path='/home/mk/gaussian_swarm/gauss_comp/out_ionized.cube')
field.set_origin(np.array([6.36, 14.86, 2.75]))

h_chain.add_field(field, eps=1)

num_sites = h_0.shape[0]
num_sites1 = 2 * num_periods * num_sites + num_sites
num_periods = 2 * num_periods + 1

dos1 = np.zeros((energy.shape[0]))

for j, E in enumerate(energy):

    h_chain.add_self_energies(sgf_l[j, :, :], sgf_r[j, :, :])
    grd, grl, gru, gr_left = recursive_gf(E, h_chain.h_l, h_chain.h_0, h_chain.h_r)
    h_chain.remove_self_energies()

    print("{} of {}: energy is {}".format(j + 1, energy.shape[0], E))

    for jj in range(len(grd)):
        dos1[j] = dos1[j] + np.real(np.trace(1j * (grd[jj] - grd[jj].H))) / num_periods

print('hi')
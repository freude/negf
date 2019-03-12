import sys
sys.path.append('./negf')
sys.path.append('../negf')
import numpy as np
import matplotlib.pyplot as plt
from negf.hamiltonian_chain import HamiltonianChain
from negf.field import Field
from negf.recursive_greens_functions import recursive_gf


def generate_hamiltonian():

    h_l = np.load('h_l.npy')
    h_0 = np.load('h_0.npy')
    h_r = np.load('h_r.npy')
    coords = np.load('coords.npy')

    num_sites = h_0.shape[0]

    field = Field(path='/home/mk/gaussian_swarm/gauss_comp/out_ionized.cube')
    field.set_origin(np.array([6.36, 11.86, 2.75]))

    h = HamiltonianChain(h_l, h_0, h_r, coords)

    period = np.array([0, 0, 5.50])
    h.translate(period, 2, 2)
    # h.add_field(field, eps=1)

    return h.h_l, h.h_0, h.h_r


sgf_l = np.load('sgf_l.npy')
sgf_r = np.load('sgf_r.npy')

energy = np.linspace(2.1, 2.2, 50)

tr = np.zeros((energy.shape[0]), dtype=np.complex)
dos = np.zeros((energy.shape[0]), dtype=np.complex)

mat_l_list, mat_d_list, mat_u_list = generate_hamiltonian()

for j, E in enumerate(energy):
    print(j)

    mat_d_list[0] = mat_d_list[0] + sgf_l[j, :, :]
    mat_d_list[-1] = mat_d_list[-1] + sgf_r[j, :, :]

    grd, grl, gru, gr_left = recursive_gf(E, mat_d_list, mat_l_list, mat_u_list)

    mat_d_list[0] = mat_d_list[0] - sgf_l[j, :, :]
    mat_d_list[-1] = mat_d_list[-1] - sgf_r[j, :, :]

    for jj in range(len(grd)):
        dos[j] += np.real(np.trace(1j * (grd[jj] - grd[jj].H)))


plt.plot(tr)
plt.show()
plt.plot(dos)
plt.show()

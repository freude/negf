import numpy as np
import matplotlib.pyplot as plt
from .field import Field
from .recursive_greens_functions import recursive_gf


def generate_hamiltonian():

    h_l = np.load('h_l.npy')
    h_0 = np.load('h_0.npy')
    h_r = np.load('h_r.npy')
    coords = np.load('coords.npy')

    num_sites = h_0.shape[0]

    field = Field(path='/home/mk/gaussian_swarm/gauss_comp/out_ionized.cube')
    field.set_origin(np.array([6.36, 11.86 + 10, 2.75]))

    period = np.array([0, 0, 5.50])
    values_m2 = field.get_values(coords, translate=-2 * period)
    values_m1 = field.get_values(coords, translate=-period)
    values_0 = field.get_values(coords)
    values_1 = field.get_values(coords, translate=period)
    values_2 = field.get_values(coords, translate=2 * period)

    eps = 7.0

    mat_d_list = [h_0+0.0*values_m2,
                  h_0+0.0*values_m2,
                  h_0+0.0*values_m1,
                  h_0+0.0*values_0,
                  h_0+0.0*values_1,
                  h_0+0.0*values_2,
                  h_0+0.0*values_2]

    mat_u_list = [h_l, h_l, h_l, h_l, h_l, h_l]
    mat_l_list = [h_r, h_r, h_r, h_r, h_r, h_r]



if __name__ == '__main__':



    sgf_l = np.load('sgf_l.npy')
    sgf_r = np.load('sgf_r.npy')

    energy = np.linspace(2.1, 2.2, 50)

    tr = np.zeros((energy.shape[0]), dtype=np.complex)
    dos = np.zeros((energy.shape[0]), dtype=np.complex)



    for j, E in enumerate(energy):
        print(j)

        mat_d_list[0] += sgf_l[j, :, :]
        mat_d_list[-1] += sgf_r[j, :, :]

        grd, grl, gru, gr_left = recursive_gf(E, mat_d_list, mat_u_list, mat_l_list)

        mat_d_list[0] -= sgf_l[j, :, :]
        mat_d_list[-1] -= sgf_r[j, :, :]

        for jj in range(len(grd)):
            dos[j] += np.real(np.trace(1j * (grd[jj] - grd[jj].H)))

    plt.plot(dos)
    plt.show()
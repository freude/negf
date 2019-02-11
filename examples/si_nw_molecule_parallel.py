import os
import shutil
from mpi4py import MPI
import numpy as np
import tb
from negf.hamiltonian_chain import HamiltonianChain
from negf.field import Field
from negf.recursive_greens_functions import recursive_gf


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

save_to = './SiNW/'


def compute_tb_matrices(input_file, save=save_to):
    """
    The script computes or load TB matrices from the disk.

    :param input_file:   xyz-file for atomic system
    :param save:         if is not None, save TB matrices into the directory specified by the parameter save
    :return:             h_l, h_0, h_r, coords, path
    """

    if os.path.isdir(input_file):
        h_l = np.load(os.path.join(input_file, 'h_l.npy'))
        h_0 = np.load(os.path.join(input_file, 'h_0.npy'))
        h_r = np.load(os.path.join(input_file, 'h_r.npy'))
        coords = np.load(os.path.join(input_file, 'coords.npy'))
        path = input_file
    else:
        tb.Atom.orbital_sets = {'Si': 'SiliconSP3D5S', 'H': 'HydrogenS'}
        h = tb.Hamiltonian(xyz=input_file, nn_distance=2.4)
        h.initialize()
        period = [0, 0, 5.50]
        h.set_periodic_bc([period])
        h_l, h_0, h_r = h.get_coupling_hamiltonians()

        coords = h.get_site_coordinates()

        if save:
            sys_name = os.path.basename(input_file)  # get the base name
            sys_name = os.path.splitext(sys_name)[0]  # get rid of the extension
            path = os.path.join(save, sys_name)

            shutil.copy2(input_file, path)

            np.save(os.path.join(path, 'h_l'), h_l)
            np.save(os.path.join(path, 'h_0'), h_0)
            np.save(os.path.join(path, 'h_r'), h_r)
            np.save(os.path.join(path, 'coords'), coords)
        else:
            path = None

    return h_l, h_0, h_r, coords, path


def compute_self_energies_for_leads(energy, h_l, h_0, h_r, save=None):

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
        np.save(os.path.join(save, 'sgf_l'), sgf_l)
        np.save(os.path.join(save, 'sgf_r'), sgf_r)
        np.save(os.path.join(save, 'energy'), energy)

    return sgf_l, sgf_r


h_l, h_0, h_r, coords, path = compute_tb_matrices(input_file='./SiNW/SiNW2/')
# h_l, h_0, h_r, coords, path = compute_tb_matrices(input_file='/home/mk/TB_project/tb/third_party/SiNW6.xyz')
# h_l, h_0, h_r, coords, path = compute_tb_matrices(input_file='/home/mk/TB_project/tb/third_party/SiNW7.xyz')

energy = np.linspace(2.1, 2.2, 50)
energy = energy[:20]

# sgf_l, sgf_r = compute_self_energies_for_leads(energy, h_l, h_0, h_r, save='./SiNW/SiNW2/')
# sgf_l = np.load('sgf_l.npy')
# sgf_r = np.load('sgf_r.npy')

num_periods = 3

h_chain = HamiltonianChain(h_l, h_0, h_r, coords)
h_chain.translate([[0, 0, 5.50]], num_periods, num_periods)

# field = Field(path='/home/mk/gaussian_swarm/gauss_comp/out_ionized.cube')
# field.set_origin(np.array([6.36, 12.86, 2.75]))

# h_chain.add_field(field, eps=1)

num_sites = h_0.shape[0]
num_sites1 = 2 * num_periods * num_sites + num_sites
num_periods = 2 * num_periods + 1

dos1 = np.zeros((energy.shape[0]))
dos = []

for j, E in enumerate(energy):
    if j % size != rank:
        continue

    L, R = tb.surface_greens_function(E, h_l, h_0, h_r, iterate=True)
    h_chain.add_self_energies(L, R)
    grd, grl, gru, gr_left = recursive_gf(E, h_chain.h_l, h_chain.h_0, h_chain.h_r)
    h_chain.remove_self_energies()

    print("{} of {}: energy is {}".format(j + 1, energy.shape[0], E))

    for jj in range(len(grd)):
        dos1[j] = dos1[j] + np.real(np.trace(1j * (grd[jj] - grd[jj].H))) / num_periods

    dos.append({'id': j, 'dos': dos1[j]})

dos = comm.reduce(dos, root=0)

print('hi')
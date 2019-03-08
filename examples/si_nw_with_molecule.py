import os
import shutil
import numpy as np
import tb
from negf.hamiltonian_chain import HamiltonianChain, HamiltonianChainComposer
from negf.field import Field
from negf.recursive_greens_functions import recursive_gf
from ase.visualize.plot import plot_atoms, Matplotlib
from ase.io import read
from negf.aux_functions import yaml_parser


save_to = './SiNW/'


def se(energy, e1, e2):
    """
    Simplest self-energy defined as a non-zero complex number in a range of energies (e1, e2)

    :param energy:
    :param e1:
    :param e2:
    :return:
    """

    ans = 0

    if e1 < energy < e2:
        ans = -0.01j

    ans = -0.007j

    return ans


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


def visualize(hc, field, size_x_min, size_y_min, size_z_min):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib import cm

    # ----------------------------------------------------------------------------

    slab = read('./SiNW/SiNW2/SiNW2f.xyz', format='xyz')

    # ----------------------------------------------------------------------------

    x = np.linspace(-14, 26, 100)
    y = np.linspace(-3, 30, 100)
    z = np.linspace(-20, 20, 100)

    X = np.meshgrid(x, y, z, indexing='ij')
    data = field.get_values(np.vstack((X[0].flatten(),
                                       X[1].flatten(),
                                       X[2].flatten())).T).reshape(X[0].shape) / 3.8

    # ----------------------------------------------------------------------------

    cut_level = 0.01*20
    data[data > cut_level] = cut_level
    # data[data < -cut_level] = -cut_level

    # ----------------------------------------------------------------------------

    n_contours = 21

    norm = cm.colors.Normalize(vmax=cut_level, vmin=-cut_level)
    # cmap = cm.PRGn
    # cmap = cm.seismic
    cmap = cm.coolwarm
    levels = np.arange(-cut_level * 1.1, cut_level * 1.1, 2 * cut_level / n_contours)

    # ----------------------------------------------------------------------------

    x = (x, y, z)

    mins = [max(np.min(hc.coords[:, j]), np.min(x[j])) for j in range(3)]
    sizes = [min(np.max(hc.coords[:, j]) - mins[j], np.max(x[j]) - mins[j]) for j in range(3)]
    # inds = [np.argmin(np.abs(x + field._origin_shift[j])) for j in range(3)]
    inds = [np.argmin(np.abs(x[j] - mins[j] - 0.5 * sizes[j])) for j in range(3)]

    # ----------------------------------------------------------------------------

    # fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex='all', sharey='all')
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    set_ind = {0, 1, 2}

    shift = 100

    for j1 in range(3):
        for j2 in range(3):
            if j1 != j2 and j2 > j1:

                if j1 == 1:
                    jj2 = j1
                    jj1 = j2
                else:
                    jj1 = j1
                    jj2 = j2

                j = (set_ind - {j1, j2}).pop()

                cset = ax[max(j2 - j1 - 1, 0), j1].contourf(np.take(X[jj1], inds[j], j)+shift,
                                                            np.take(X[jj2], inds[j], j)+shift,
                                                            np.take(data, inds[j], j),
                                                            levels,
                                                            norm=norm,
                                                            cmap=cm.get_cmap(cmap, len(levels) - 1))

                cset = ax[max(j2 - j1 - 1, 0), j1].contour(np.take(X[jj1], inds[j], j)+shift,
                                                           np.take(X[jj2], inds[j], j)+shift,
                                                           np.take(data, inds[j], j),
                                                           levels,
                                                           norm=norm,
                                                           colors='k',
                                                           linewidths=1)

                # ax[max(j2 - j1 - 1, 0), j1].add_patch(Rectangle((mins[jj1]+shift, mins[jj2]+shift),
                #                                                 sizes[jj1],
                #                                                 sizes[jj2],
                #                                                 alpha=1,
                #                                                 fill=None))

                radii = 0.5

                if j == 0:
                    rotation = ('0x,90y,0z')
                    offsets = (size_z_min+shift+radii-5, size_y_min+shift-radii)
                elif j == 1:
                    rotation = ('90x,0y,0z')
                    offsets = (size_x_min+shift-radii, size_z_min+shift+radii-5)
                else:
                    rotation = ('0x,0y,0z')
                    offsets = (size_x_min+shift-radii, size_y_min+shift-radii)

                plot_atoms(slab, ax=ax[max(j2 - j1 - 1, 0), j1], radii=radii, offset=offsets, rotation=rotation)

                ax[max(j2 - j1 - 1, 0), j1].axis('off')
                ax[max(j2 - j1 - 1, 0), j1].set_ylim(np.min(x[jj2])+shift, np.max(x[jj2])+shift)

                if j1 != jj1:
                    ax[max(j2 - j1 - 1, 0), j1].set_xlim(np.max(x[jj1])+shift, np.min(x[jj1])+shift)
                else:
                    ax[max(j2 - j1 - 1, 0), j1].set_xlim(np.min(x[jj1])+shift, np.max(x[jj1])+shift)

    ax[1, 1].axis('off')
    plt.tight_layout()
    plt.show()


def visualize1(hc, field, size_x_min, size_y_min, size_z_min):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib import cm

    # ----------------------------------------------------------------------------

    slab = read('./SiNW/SiNW2/SiNW2f.xyz', format='xyz')

    # ----------------------------------------------------------------------------

    x = np.linspace(-14, 26, 100)
    y = np.linspace(-3, 30, 100)
    z = np.linspace(-20, 20, 100)

    X = np.meshgrid(x, y, z, indexing='ij')
    data = field.get_values(np.vstack((X[0].flatten(),
                                       X[1].flatten(),
                                       X[2].flatten())).T).reshape(X[0].shape) / 3.8

    # ----------------------------------------------------------------------------

    cut_level = 0.01*20
    data[data > cut_level] = cut_level
    # data[data < -cut_level] = -cut_level

    # ----------------------------------------------------------------------------

    n_contours = 21

    norm = cm.colors.Normalize(vmax=cut_level, vmin=-cut_level)
    # cmap = cm.PRGn
    # cmap = cm.seismic
    cmap = cm.coolwarm
    levels = np.arange(-cut_level * 1.1, cut_level * 1.1, 2 * cut_level / n_contours)

    # ----------------------------------------------------------------------------

    x = (x, y, z)

    mins = [max(np.min(hc.coords[:, j]), np.min(x[j])) for j in range(3)]
    sizes = [min(np.max(hc.coords[:, j]) - mins[j], np.max(x[j]) - mins[j]) for j in range(3)]
    # inds = [np.argmin(np.abs(x + field._origin_shift[j])) for j in range(3)]
    inds = [np.argmin(np.abs(x[j] - mins[j] - 0.5 * sizes[j])) for j in range(3)]

    # ----------------------------------------------------------------------------

    set_ind = {0, 1, 2}

    shift = 100

    for j1 in range(3):
        for j2 in range(3):
            if j1 != j2 and j2 > j1:

                if j1 == 1:
                    jj2 = j1
                    jj1 = j2
                else:
                    jj1 = j1
                    jj2 = j2

                j = (set_ind - {j1, j2}).pop()

                fig, ax = plt.subplots(figsize=(10, 10))

                cset = ax.contourf(np.take(X[jj1], inds[j], j)+shift,
                                                            np.take(X[jj2], inds[j], j)+shift,
                                                            np.take(data, inds[j], j),
                                                            levels,
                                                            norm=norm,
                                                            cmap=cm.get_cmap(cmap, len(levels) - 1))

                cset = ax.contour(np.take(X[jj1], inds[j], j)+shift,
                                                           np.take(X[jj2], inds[j], j)+shift,
                                                           np.take(data, inds[j], j),
                                                           levels,
                                                           norm=norm,
                                                           colors='k',
                                                           linewidths=1)

                # ax[max(j2 - j1 - 1, 0), j1].add_patch(Rectangle((mins[jj1]+shift, mins[jj2]+shift),
                #                                                 sizes[jj1],
                #                                                 sizes[jj2],
                #                                                 alpha=1,
                #                                                 fill=None))

                radii = 0.5

                if j == 0:
                    rotation = ('0x,90y,0z')
                    offsets = (size_z_min+shift+radii-5, size_y_min+shift-radii)
                elif j == 1:
                    rotation = ('90x,0y,0z')
                    offsets = (size_x_min+shift-radii, size_z_min+shift+radii-5)
                else:
                    rotation = ('0x,0y,0z')
                    offsets = (size_x_min+shift-radii, size_y_min+shift-radii)

                plot_atoms(slab, ax=ax, radii=radii, offset=offsets, rotation=rotation)

                ax.axis('off')
                ax.set_ylim(np.min(x[jj2])+shift, np.max(x[jj2])+shift)

                if j1 != jj1:
                    ax.set_xlim(np.max(x[jj1])+shift, np.min(x[jj1])+shift)
                else:
                    ax.set_xlim(np.min(x[jj1])+shift, np.max(x[jj1])+shift)

                plt.tight_layout()
                plt.show()


def main(spacing, mol_path, nw_path, eps, comm=0):

    if comm:
        rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        rank = 0
        size = 1

    # ---------------------------------------------------------------------------------
    # ----------- compute tight-binding matrices and define energy scale --------------
    # ---------------------------------------------------------------------------------

    h_l, h_0, h_r, coords, path = compute_tb_matrices(input_file=nw_path)
    energy = np.linspace(2.1, 2.15, 50)
    energy = energy[15:30]

    # ---------------------------------------------------------------------------------
    # ------- pre-compute/pre-load self-energies for the leads from the disk ----------
    # ---------------------------------------------------------------------------------

    # sgf_l, sgf_r = compute_self_energies_for_leads(energy, h_l, h_0, h_r, save='./SiNW/SiNW2/')
    # sgf_l = np.load('sgf_l.npy')
    # sgf_r = np.load('sgf_r.npy')

    # ---------------------------------------------------------------------------------
    # ---------------------------- make a chain Hamiltonian ---------------------------
    # ---------------------------------------------------------------------------------

    num_periods = 3  # number of unit cells in the device region num_periods * 2 + 1

    h_chain = HamiltonianChain(h_l, h_0, h_r, coords)
    h_chain.translate([[0, 0, 5.50]], num_periods, num_periods)

    # ---------------------------------------------------------------------------------
    # --------------------- make a Field object from the cube file --------------------
    # ---------------------------------------------------------------------------------

    field = Field(path=mol_path)

    angle = 1.13446                    # 65 degrees
    field.rotate('x', angle)
    field.rotate('y', np.pi / 2.0)

    # field.set_origin(np.array([6.36, 11.86, 2.75]))
    # field.set_origin(np.array([-11.82 - 11.5, 0.0, 5.91]))

    size_x_min = np.min(coords[:, 0])
    size_x_max = np.max(coords[:, 0])
    size_y_min = np.min(coords[:, 1])
    size_y_max = np.max(coords[:, 1])
    size_z_min = -np.max(coords[:, 2]) * 4
    size_z_max = np.max(coords[:, 2]) * 3

    _, mol_coords = field.get_atoms()
    mol_y_length = np.max(mol_coords[:, 1]) - np.min(mol_coords[:, 1])
    mol_y_length = mol_y_length * np.sin(angle)
    mol_z_length = mol_y_length * np.cos(angle)

    field.set_origin(np.array([0.5 * (size_x_max - np.abs(size_y_min)),
                               size_y_max + 0.5 * mol_y_length + spacing,
                               0.5*mol_z_length]))

    # ---------------------------------------------------------------------------------
    # ------------------- add field to the Hamiltonian and visualize ------------------
    # ---------------------------------------------------------------------------------

    h_chain.add_field(field, eps=eps)
    h_chain.visualize()
    # visualize1(h_chain, field, size_x_min, size_y_min, size_z_min)

    # ---------------------------------------------------------------------------------
    # -------------------- compute Green's functions of the system --------------------
    # ---------------------------------------------------------------------------------

    num_periods = 2 * num_periods + 1

    dos = np.zeros((energy.shape[0]))
    tr = np.zeros((energy.shape[0]))
    dens = np.zeros((energy.shape[0], num_periods))

    par_data = []

    ef1 = 2.1
    ef2 = 2.1
    tempr = 100

    for j, E in enumerate(energy):
        if j % size != rank:
            continue

        L, R = tb.surface_greens_function(E, h_l, h_0, h_r, iterate=5)

        # L = L + se(E, 2.0, 2.125)
        # R = R + se(E, 2.0, 2.125)

        h_chain.add_self_energies(L, R, energy=E, tempr=tempr, ef1=ef1, ef2=ef2)
        g_trans, grd, grl, gru, gr_left, gnd, gnl, gnu, gn_left = recursive_gf(E,
                                                                      h_chain.h_l,
                                                                      h_chain.h_0,
                                                                      h_chain.h_r,
                                                                      s_in=h_chain.sgf)
        h_chain.remove_self_energies()

        for jj in range(num_periods):
            dos[j] = dos[j] + np.real(np.trace(1j * (grd[jj] - grd[jj].H))) / num_periods
            dens[j, jj] = 2 * np.trace(gnd[jj]) / num_periods

        gamma_l = 1j * (np.matrix(L) - np.matrix(L).H)
        gamma_r = 1j * (np.matrix(R) - np.matrix(R).H)
        tr[j] = np.real(np.trace(gamma_l * g_trans * gamma_r * g_trans.H))

        print("{} of {}: energy is {}".format(j + 1, energy.shape[0], E))

        if comm:
            par_data.append({'id': j, 'dos': dos[j], 'tr': tr[j], 'dens': dens[j]})

    if comm:
        par_data = comm.reduce(par_data, root=0)
        if rank == 0:
            ids = [par_data[item]['id'] for item in range(len(par_data))]
            dos = [x['dos'] for _, x in sorted(zip(ids, par_data))]
            tr = [x['tr'] for _, x in sorted(zip(ids, par_data))]
            dens = [x['dens'] for _, x in sorted(zip(ids, par_data))]
            dos = np.array(dos)
            tr = np.array(tr)
            dens = np.array(dens)

            # np.save('dos.npy', dos)

    return dos, tr, dens


def main1(nw_path, fields_config, comm=0):

    if comm:
        rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        rank = 0
        size = 1

    params = yaml_parser(fields_config)

    # ---------------------------------------------------------------------------------
    # ------------compute tight-binding matrices and define energy scale --------------
    # ---------------------------------------------------------------------------------

    h_l, h_0, h_r, coords, path = compute_tb_matrices(input_file=nw_path)

    energy = np.linspace(2.1, 2.15, 50)

    # ---------------------------------------------------------------------------------
    # ------- pre-compute/pre-load self-energies for the leads from the disk ----------
    # ---------------------------------------------------------------------------------

    # sgf_l, sgf_r = compute_self_energies_for_leads(energy, h_l, h_0, h_r, save='./SiNW/SiNW2/')
    # sgf_l = np.load('sgf_l.npy')
    # sgf_r = np.load('sgf_r.npy')

    # ---------------------------------------------------------------------------------
    # ---------------------------- make a chain Hamiltonian ---------------------------
    # ---------------------------------------------------------------------------------

    h_chain = HamiltonianChainComposer(h_l, h_0, h_r, coords, params)
    h_chain.visualize()

    # ---------------------------------------------------------------------------------
    # -------------------- compute Green's functions of the system --------------------
    # ---------------------------------------------------------------------------------

    num_periods = params['left_translations'] + params['right_translations'] + 1

    dos = np.zeros((energy.shape[0]))
    tr = np.zeros((energy.shape[0]))
    dens = np.zeros((energy.shape[0], num_periods))

    par_data = []

    ef1 = 2.1
    ef2 = 2.1
    tempr = 100

    for j, E in enumerate(energy):
        if j % size != rank:
            continue

        L, R = tb.surface_greens_function(E, h_l, h_0, h_r, iterate=5)

        # L = L + se(E, 2.0, 2.125)
        # R = R + se(E, 2.0, 2.125)

        h_chain.add_self_energies(L, R, energy=E, tempr=tempr, ef1=ef1, ef2=ef2)
        g_trans, grd, grl, gru, gr_left, gnd, gnl, gnu, gn_left = recursive_gf(E,
                                                                               h_chain.h_l,
                                                                               h_chain.h_0,
                                                                               h_chain.h_r,
                                                                               s_in=h_chain.sgf)
        h_chain.remove_self_energies()

        for jj in range(num_periods):
            dos[j] = dos[j] + np.real(np.trace(1j * (grd[jj] - grd[jj].H))) / num_periods
            dens[j, jj] = 2 * np.trace(gnd[jj]) / num_periods

        gamma_l = 1j * (np.matrix(L) - np.matrix(L).H)
        gamma_r = 1j * (np.matrix(R) - np.matrix(R).H)
        tr[j] = np.real(np.trace(gamma_l * g_trans * gamma_r * g_trans.H))

        print("{} of {}: energy is {}".format(j + 1, energy.shape[0], E))

        if comm:
            par_data.append({'id': j, 'dos': dos[j], 'tr': tr[j], 'dens': dens[j]})

    if comm:

        par_data = comm.reduce(par_data, root=0)

        if rank == 0:
            ids = [par_data[item]['id'] for item in range(len(par_data))]
            dos = [x['dos'] for _, x in sorted(zip(ids, par_data))]
            tr = [x['tr'] for _, x in sorted(zip(ids, par_data))]
            dens = [x['dens'] for _, x in sorted(zip(ids, par_data))]
            dos = np.array(dos)
            tr = np.array(tr)
            dens = np.array(dens)

            np.save('dos' + params['job_title'] + '.npy', dos)
            np.save('tr' + params['job_title'] + '.npy', tr)
            np.save('dens' + params['job_title'] + '.npy', dens)

            return dos, tr, dens

    else:
        np.save('dos' + params['job_title'] + '.npy', dos)
        np.save('tr' + params['job_title'] + '.npy', tr)
        np.save('dens' + params['job_title'] + '.npy', dens)

        return dos, tr, dens


if __name__ == '__main__':

    # main(spacing=1.0,
    #      mol_path='/home/mk/tetracene_dft_wB_pcm_38_32_cation.cube',
    #      nw_path='./SiNW/SiNW2/',
    #      eps=3.8)

    fields_config = """

    job_title:              '1'

    unit_cell:        [[0, 0, 5.50]]

    left_translations:     3
    right_translations:    3

    fields:

        eps:          3.8

        cation:      "/home/mk/tetracene_dft_wB_pcm_38_32_cation.cube"

        angle:       1.13446
        spacing:     3.0

        xyz:
            - cation:       [-5.0000000000,    0.0000000000,    -5.0000000000]
            - cation:       [5.0000000000,    0.0000000000,    5.0000000000]
        """

    main1(nw_path='./SiNW/SiNW2/', fields_config=fields_config)

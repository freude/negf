import copy
import numpy as np
import matplotlib.pyplot as plt


def mat_left_div(mat_a, mat_b):

    mat_a = np.asmatrix(mat_a)
    mat_b = np.asmatrix(mat_b)

    ans, resid, rank, s = np.linalg.lstsq(mat_a, mat_b, rcond=-1)

    return ans


def mat_mul(list_of_matrices):

    num_of_mat = len(list_of_matrices)

    unity = np.eye(list_of_matrices[num_of_mat - 1].shape[0])

    for j, item in enumerate(list_of_matrices):
        list_of_matrices[j] = np.matrix(item)

    for j in range(9, -1, -1):
        unity = list_of_matrices[j] * unity

    return unity


def recursive_gf(energy, mat_l_list, mat_d_list, mat_u_list):
    """
    The recursive Green's function algorithm is taken from
    M. P. Anantram, M. S. Lundstrom and D. E. Nikonov, Proceedings of the IEEE, 96, 1511 - 1550 (2008)
    DOI: 10.1109/JPROC.2008.927355


    :param energy:                     energy
    :param mat_d_list:                 list of diagonal blocks
    :param mat_u_list:                 list of upper-diagonal blocks
    :param mat_l_list:                 list of lower-diagonal blocks

    :return grd, grl, gru, gr_left:    retarded Green's
                                       function: block-diagonal,
                                                 lower block-diagonal,
                                                 upper block-diagonal,
                                                 left-connected
    """

    # convert input arrays to matrix data type
    for jj, item in enumerate(mat_d_list):
        mat_d_list[jj] = np.asmatrix(item)
        mat_d_list[jj] = mat_d_list[jj] - np.diag(energy * np.ones(mat_d_list[jj].shape[0]))

        if jj < len(mat_d_list) - 1:
            mat_u_list[jj] = np.asmatrix(mat_u_list[jj])
            mat_l_list[jj] = np.asmatrix(mat_l_list[jj])

    # computes matrix sizes
    num_of_matrices = len(mat_d_list)                  # Number of diagonal blocks.
    mat_shapes = [item.shape for item in mat_d_list]   # This gives the sizes of the diagonal matrices.

    # allocate empty lists of certain lengths
    gr_left = [None for _ in range(num_of_matrices)]

    gr_left[0] = mat_left_div(-mat_d_list[0], np.eye(mat_shapes[0][0]))   # Initialising the retarded left connected.

    for q in range(num_of_matrices - 1):                              # Recursive algorithm (B2)
        gr_left[q + 1] = mat_left_div((-mat_d_list[q + 1] - mat_l_list[q] * gr_left[q] * mat_u_list[q]),
                                      np.eye(mat_shapes[q + 1][0]))   # The left connected recursion.

    grl = [None for _ in range(num_of_matrices)]
    gru = [None for _ in range(num_of_matrices)]
    grd = copy.copy(gr_left)  # Our glorious benefactor.

    for q in range(num_of_matrices - 2, -1, -1):                    # Recursive algorithm
        grl[q] = grd[q + 1] * mat_l_list[q] * gr_left[q]           # (B5) We get the off-diagonal blocks for free.
        gru[q] = gr_left[q] * mat_u_list[q] * grd[q + 1]           # (B6)because we need .Tthem.T for the next calc:
        grd[q] = gr_left[q] + gr_left[q] * mat_u_list[q] * grl[q]   # (B4)I suppose I could also use the lower.

    for jj, item in enumerate(mat_d_list):
        mat_d_list[jj] = mat_d_list[jj] + np.diag(energy * np.ones(mat_d_list[jj].shape[0]))

    return grd, grl, gru, gr_left




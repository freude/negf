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


# def recursive_gf(mat_d_list, mat_u_list, mat_l_list, sigma_in):
#     """
#     :param mat_d_list:           list of diagonal blocks
#     :param mat_u_list:           list of upper-diagonal blocks
#     :param mat_l_list:           list of lower-diagonal blocks
#     :param sigma_in:     self_energy matrix
#
#     :return:
#     """
#
#     # convert input arrays to matrix data type
#     for jj, item in enumerate(mat_d_list):
#         mat_d_list[jj] = np.asmatrix(item)
#
#     for jj, item in enumerate(mat_u_list):
#         mat_u_list[jj] = np.asmatrix(item)
#
#     for jj, item in enumerate(mat_l_list):
#         mat_l_list[jj] = np.asmatrix(item)
#
#     sigma_in = np.asmatrix(sigma_in)
#
#     # computes matrix sizes
#     num_of_matrices = len(mat_d_list)                               # Number of diagonal blocks.
#     mat_shapes = [item.shape for item in mat_d_list]                # This gives the sizes of the diagonal matrices.
#
#     # allocate empty lists of certain lengths
#     gr_left = [None for _ in range(num_of_matrices)]
#     gn_left = [None for _ in range(num_of_matrices)]
#
#     gr_left[0] = mat_left_div(mat_d_list[0], np.eye(mat_shapes[0][0]))    # Initialising the retarded left connected.
#
#     for q in range(num_of_matrices-1):                                    # Recursive algorithm (B2)
#         gr_left[q+1] = mat_left_div((mat_d_list[q+1] - mat_l_list[q] * gr_left[q] * mat_u_list[q]),
#                                     np.eye(mat_shapes[q+1][0]))    # The left connected recursion.
#
#     grl = [None for _ in range(num_of_matrices)]
#     gru = [None for _ in range(num_of_matrices)]
#     grd = copy.copy(gr_left)                                    # Our glorious benefactor.
#
#     for q in range(num_of_matrices-2, -1, -1):                  # Recursive algorithm
#         grl[q] = -grd[q+1] * mat_l_list[q] * gr_left[q]         # (B5) We get the off-diagonal blocks for free.
#         gru[q] = -gr_left[q] * mat_u_list[q] * grd[q+1]         # (B6)because we need .Tthem.T for the next calc:
#         grd[q] = gr_left[q] - gr_left[q] * mat_u_list[q] * grl[q]   # (B4)I suppose I could also use the lower.
#
#     gn_left[0] = gr_left[0] * sigma_in[0] * gr_left[0].T
#
#     for q in range(num_of_matrices-1):  #(B8)
#         gn_left[q+1] = gr_left[q+1] * (sigma_in[q+1] + mat_l_list[q] * gn_left[q] * mat_l_list[q].T) * gr_left[q+1].T
#
#     gnd = copy.copy(gn_left)
#     gnl = [0 for _ in range(num_of_matrices - 1)]
#     gnu = [0 for _ in range(num_of_matrices - 1)]
#
#     for q in range(num_of_matrices-2, -1, -1):
#         gnl[q] = -grd[q+1]*mat_l_list[q]*gn_left[q] - gnd[q+1]*mat_u_list[q].T*gr_left[q].T  #(B10) TYPO in derivation. mat_u_list[q].T not mat_l_list[q].T
#         gnd[q] = gn_left[q] + gr_left[q]*(mat_u_list[q]*gnd[q+1]*mat_u_list[q].T)*gr_left[q].T - (gn_left[q]*mat_l_list[q].T*gru[q].T + gru[q]*mat_l_list[q]*gn_left[q]) #mat_u_list[q].T not mat_l_list[q].T in second bracket
#               #(B11) #Could do (gr_left[q]*mat_u_list[q]).T
#
#     gpd = [0 for _ in range(num_of_matrices)]
#     gpu = [0 for _ in range(num_of_matrices - 1)]
#     gpl = [0 for _ in range(num_of_matrices - 1)]
#
#     for ip in range(num_of_matrices-1):
#         gnu[ip] = gnl[ip].T
#         gpd[ip] = 1j*(grd[ip] - grd[ip].T) - gnd[ip]
#         gpu[ip] = 1j*(gru[ip] - grl[ip].T) - gnu[ip]
#         gpl[ip] = 1j*(grl[ip] - gru[ip].T) - gnl[ip]
#
#     gpd[num_of_matrices] = 1j * (grd[num_of_matrices] - grd[num_of_matrices].T) - gnd[num_of_matrices]
#
#     return grd, grl, gru, gnd, gnl, gnu, gpd, gpl, gpu, gr_left, gn_left

def recursive_gf(energy, mat_d_list, mat_u_list, mat_l_list):
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

    di = np.diag_indices(mat_d_list[0].shape[0])

    # convert input arrays to matrix data type
    for jj, item in enumerate(mat_d_list):
        mat_d_list[jj] = np.asmatrix(item)
        mat_d_list[jj][di] -= energy

    for jj, item in enumerate(mat_u_list):
        mat_u_list[jj] = np.asmatrix(item)

    for jj, item in enumerate(mat_l_list):
        mat_l_list[jj] = np.asmatrix(item)

    # computes matrix sizes
    num_of_matrices = len(mat_d_list)                  # Number of diagonal blocks.
    mat_shapes = [item.shape for item in mat_d_list]   # This gives the sizes of the diagonal matrices.

    # allocate empty lists of certain lengths
    gr_left = [None for _ in range(num_of_matrices)]

    gr_left[0] = -mat_left_div(mat_d_list[0], np.eye(mat_shapes[0][0]))   # Initialising the retarded left connected.

    for q in range(num_of_matrices - 1):                              # Recursive algorithm (B2)
        gr_left[q + 1] = mat_left_div((-mat_d_list[q + 1] - mat_l_list[q] * gr_left[q] * mat_u_list[q]),
                                      np.eye(mat_shapes[q + 1][0]))   # The left connected recursion.

    grl = [None for _ in range(num_of_matrices)]
    gru = [None for _ in range(num_of_matrices)]
    grd = copy.copy(gr_left)  # Our glorious benefactor.

    for q in range(num_of_matrices - 2, -1, -1):                    # Recursive algorithm
        grl[q] = -grd[q + 1] * mat_l_list[q] * gr_left[q]           # (B5) We get the off-diagonal blocks for free.
        gru[q] = -gr_left[q] * mat_u_list[q] * grd[q + 1]           # (B6)because we need .Tthem.T for the next calc:
        grd[q] = gr_left[q] - gr_left[q] * mat_u_list[q] * grl[q]   # (B4)I suppose I could also use the lower.

    for jj, item in enumerate(mat_d_list):
        mat_d_list[jj][di] += energy

    return grd, grl, gru, gr_left




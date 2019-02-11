import numpy as np
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
from .field import Field
from .recursive_greens_functions import recursive_gf


def fd(energy, ef, temp):
    kb = 8.61733e-5       # Boltzmann constant in eV
    return 1.0 / (1.0 + np.exp((energy - ef) / (kb * temp)))


class HamiltonianChain(object):

    def __init__(self, h_l, h_0, h_r, coords):

        self.h_l = h_l
        self.h_0 = h_0
        self.h_r = h_r
        self._coords = coords

        self.num_sites = h_0.shape[0]

        self.elem_length = None
        self.left_translations = None
        self.right_translations = None
        self.field = None
        self.sgf_l = None
        self.sgf_r = None

        self.energy = 0
        self.tempr = 0
        self.ef1 = 0
        self.ef2 = 0

    @property
    def sgf(self):

        sgf = [None for _ in range(len(self.h_0))]

        for jjj in range(len(self.h_0)):
            if jjj == 0:
                sgf[jjj] = -2.0 * np.matrix(np.imag(self.sgf_r) * fd(self.energy, self.ef1, self.tempr))
            elif jjj == len(self.h_0) - 1:
                sgf[jjj] = -2.0 * np.matrix(np.imag(self.sgf_l) * fd(self.energy, self.ef2, self.tempr))
            else:
                sgf[jjj] = np.matrix(np.zeros(self.h_0[jjj].shape))

        return sgf

    def translate(self, period, left_translations, right_translations):

        self.elem_length = np.array(period)
        self.left_translations = left_translations
        self.right_translations = right_translations

        self.h_l = [self.h_l for _ in range(left_translations)] + \
                   [self.h_l for _ in range(right_translations)]

        self.h_0 = [self.h_0 for _ in range(left_translations)] + \
                   [self.h_0] + \
                   [self.h_0 for _ in range(right_translations)]

        self.h_r = [self.h_r for _ in range(left_translations)] + \
                   [self.h_r for _ in range(right_translations)]

    def add_field(self, field, eps=7.0):

        self.field = []

        for jjj in range(self.left_translations, 0, -1):
            self.field.append(field.get_values(self._coords, translate=jjj * self.elem_length) / eps)

        self.field.append(field.get_values(self._coords) / eps)

        for jjj in range(1, self.right_translations + 1):
            self.field.append(field.get_values(self._coords, translate=-jjj * self.elem_length) / eps)

        for jjj in range(len(self.h_0)):
            self.h_0[jjj] = self.h_0[jjj] + np.diag(self.field[jjj])

    def remove_field(self):

        if isinstance(self.field, list):
            for jjj in range(len(self.h_0)):
                self.h_0[jjj] = self.h_0[jjj] - np.diag(self.field[jjj])

        self.field = None

    def add_self_energies(self, sgf_l, sgf_r, energy=0, tempr=0, ef1=0, ef2=0):

        self.energy = energy
        self.tempr = tempr
        self.ef1 = ef1
        self.ef2 = ef2

        self.sgf_l = sgf_l
        self.sgf_r = sgf_r

        self.h_0[-1] = self.h_0[-1] + sgf_l
        self.h_0[0] = self.h_0[0] + sgf_r

    def remove_self_energies(self):

        self.h_0[-1] = self.h_0[-1] - self.sgf_l
        self.h_0[0] = self.h_0[0] - self.sgf_r

        self.sgf_l = None
        self.sgf_r = None

    def translate_self_energies(self, sgf_l, sgf_r):

        mat_list = [item * 0.0 for item in self.h_0]

        return block_diag(*tuple([sgf_r] + mat_list[1:])), block_diag(*tuple(mat_list[:-1] + [sgf_l]))

    @property
    def coords(self):
        if self.elem_length is not None:
            coords = [self._coords - jjj * self.elem_length for jjj in range(self.left_translations, 0, -1)] + \
                     [self._coords] + \
                     [self._coords + jjj * self.elem_length for jjj in range(1, self.right_translations + 1)]

            return np.concatenate(coords)
        else:
            return self._coords

    def get_matrix(self):

        if isinstance(self.h_0, list):
            matrix = block_diag(*tuple(self.h_0))
        else:
            return self.h_0

        for j in range(len(self.h_l)):

            s1, s2 = self.h_0[j].shape

            matrix[j * s1:(j + 1) * s1, (j + 1) * s2:(j + 2) * s2] = self.h_r[j]
            matrix[(j + 1) * s1:(j + 2) * s1, j * s2:(j + 1) * s2] = self.h_l[j]

        return matrix

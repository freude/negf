import numpy as np
import matplotlib.pyplot as plt
from .field import Field
from .recursive_greens_functions import recursive_gf


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

    def translate(self, period, left_translations, right_translations):

        self.elem_length = period
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

        self.field = [field.get_values(self._coords, translate=-jjj * self.elem_length) / eps
                      for jjj in range(self.left_translations, 0, -1)] + \
                     [field.get_values(self._coords) / eps] + \
                     [field.get_values(self._coords, translate=-jjj * self.elem_length) / eps
                      for jjj in range(1, self.right_translations + 1)]

        for jjj in range(len(self.h_0)):
            self.h_0[jjj].flat[::self.h_0[jjj].shape[0] + 1] += self.field[jjj]

    def remove_field(self):

        if isinstance(self.field, list):
            for jjj in range(len(self.h_0)):
                self.h_0[jjj].flat[::self.h_0[jjj].shape[0] + 1] -= self.field[jjj]

    @property
    def coords(self):
        if self.elem_length is not None:
            coords = [self._coords - jjj * self.elem_length for jjj in range(self.left_translations, 0, -1)] + \
                     [self._coords] + \
                     [self._coords + jjj * self.elem_length for jjj in range(1, self.right_translations + 1)]

            return np.concatenate(coords)
        else:
            return self._coords

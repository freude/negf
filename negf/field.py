import numpy as np
from scipy.interpolate import RegularGridInterpolator


def _getline(cube):
    """
    Read a line from cube file where first field is an int
    and the remaining fields are floats.

    params:
        cube: file object of the cube file

    returns: (int, list<float>)
    """
    l = cube.readline().strip().split()
    return int(l[0]), map(float, l[1:])


def read_cube(fname):
    """
    Read cube file into numpy array

    params:
        fname: filename of cube file

    returns: (data: np.array, metadata: dict)
    """
    meta = {}
    with open(fname, 'r') as cube:
        cube.readline()
        cube.readline()  # ignore comments
        natm, meta['org'] = _getline(cube)
        nx, meta['xvec'] = _getline(cube)
        ny, meta['yvec'] = _getline(cube)
        nz, meta['zvec'] = _getline(cube)
        meta['atoms'] = [_getline(cube) for i in range(natm)]
        data = np.zeros((nx * ny * nz))
        idx = 0
        for line in cube:
            for val in line.strip().split():
                data[idx] = float(val)
                idx += 1
    data = np.reshape(data, (nx, ny, nz))
    return data, meta


class Field(object):

    def __init__(self, path):

        self._origin_shift = np.array([0.0, 0.0, 0.0])
        self._cube = []
        self._origin_has_changed = False

        if path.endswith('.cube') or path.endswith('.cub'):

            data, meta = read_cube(path)

            data = np.swapaxes(data, 1, 2)

            shape = data.shape
            x = np.linspace(meta['org'][0], meta['xvec'][0] * shape[0], shape[0])
            z = np.linspace(meta['org'][1], meta['yvec'][1] * shape[1], shape[1])
            y = np.linspace(meta['org'][2], meta['zvec'][2] * shape[2], shape[2])

            print(shape)
            print(meta)
            print(meta['org'])
            print(meta['xvec'])
            print(meta['yvec'])
            print(meta['zvec'])

            self._cube = [[meta['org'][0]*0.529177,
                           meta['org'][2]*0.529177,
                           meta['org'][1]*0.529177],
                          [meta['xvec'][0] * shape[0]*0.529177 - meta['org'][0]*0.529177,
                           meta['zvec'][2] * shape[2]*0.529177 - meta['org'][2]*0.529177,
                           meta['yvec'][1] * shape[1]*0.529177 - meta['org'][1]*0.529177]]

            self.origin = (np.array(self._cube[1])-np.array(self._cube[0]))*0.25
            self._shape = shape

        self._interpolant = RegularGridInterpolator((x, y, z), data, bounds_error=False)

    def set_origin(self, origin):

        if isinstance(origin, list):
            origin = np.array(origin)

        self._origin_shift = origin
        self._origin_has_changed = True

    def get_values(self, coords1, translate=None):

        coords = coords1.copy()

        if len(coords.shape) < 2:
            coords = [coords]

        for j in range(len(coords)):
            coords[j] += self.origin

            if self._origin_has_changed:
                coords[j] -= self._origin_shift

            if isinstance(translate, np.ndarray):
                coords[j] -= translate

        values = self._interpolant(coords)

        return np.nan_to_num(values)


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    fl = Field(path='/home/mk/gaussian_swarm/gauss_comp/out_neutral.cube')

    x = np.linspace(fl._cube[0][0], fl._cube[1][0], fl._shape[0])
    y = np.linspace(fl._cube[0][1], fl._cube[1][1], fl._shape[1])
    z = np.linspace(fl._cube[0][2], fl._cube[1][2], fl._shape[2])

    x = np.linspace(0, 60, 100)
    y = np.linspace(0, 11, 100)
    z = np.linspace(0, 5.5, 100)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # data = fl.get_values(np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T, translate=np.array([0,0,15]))
    # data = data.reshape(X.shape)
    # plt.imshow(data[:, :, 50])
    # # plt.contour(data[:, 50, :], levels=(0.04, 0.06, 0.08, 0.1, 0.2))
    # plt.show()

    fl.set_origin(np.array([20, 11, 2.75]))

    data = fl.get_values(np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T)
    data = data.reshape(X.shape)
    plt.imshow(data[:, :, 50])

    # plt.contour(data[:, 50, :], levels=(0.04, 0.06, 0.08, 0.1, 0.2))

    plt.show()

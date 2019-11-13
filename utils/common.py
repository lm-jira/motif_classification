import os

import h5py
import numpy as np


def load_mat(dir_path, mode):
    mat_path = os.path.join(dir_path, "{}.mat".format(mode))

    data = h5py.File(mat_path, 'r')
    label = data['{}data'.format(mode)]
    x_data = data['{}xdata'.format(mode)]

    label = np.transpose(label, (1, 0))
    x_data = np.transpose(x_data, (2, 0, 1))

    return x_data, label

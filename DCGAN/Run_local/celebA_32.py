"""
tsly, Sat Dec  8 16:34:49 2018

This code is taken from https://github.com/tatsy/keras-generative with small modifications.

"""

import h5py
import numpy as np

class Dataset(object):
    def __init__(self):
        self.images = None

    def __len__(self):
        return len(self.images)

    def _get_shape(self):
        return self.images.shape

    shape = property(_get_shape)

#def load_data(filename, size=-1):
def load_data(filename = 'celebA_32.hdf5', size=-1):
    f = h5py.File(filename)

    dset = Dataset()
    dset.images = np.asarray(f['images'], 'float32') / 255.0

    if size > 0:
        dset.images = dset.images[:size]

    return dset

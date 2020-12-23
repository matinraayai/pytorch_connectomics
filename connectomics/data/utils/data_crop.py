import numpy as np

####################################################################
## Process image stacks.
####################################################################


def count_volume(data_sz, vol_sz, stride):
    return 1 + np.ceil((data_sz - vol_sz) / stride.astype(float)).astype(int)



def crop_volume(data, size, stride=(0, 0, 0)):  # C*D*W*H, C=1
    assert hasattr(data, 'shape') and len(data.shape) == 3, 'original data has to be of a shape of length 3'
    size = np.array(size).astype(np.int)
    assert size.shape == (3,), 'size of the cropped volume has to be a list of length 3'
    stride = np.array(stride).astype(np.int)
    assert stride.shape == (3,), 'stride of the cropped volume has to be a list of length 3'
    return data[stride[0]:stride[0] + size[0], stride[1]:stride[1] + size[1], stride[2]:stride[2] + size[2]]


def crop_multi_channel_volume(data, size, stride=(0, 0, 0)):  # C*D*W*H, for multi-channel input
    assert hasattr(data, 'shape') and len(data.shape) == 4, 'original data has to be of a shape of length 4'
    size = np.array(size).astype(np.int)
    assert size.shape == (3,), 'size of the cropped volume has to be a list of length 3'
    stride = np.array(stride).astype(np.int)
    assert stride.shape == (3,), 'stride of the cropped volume has to be a list of length 3'
    return data[:, stride[0]:stride[0] + size[0], stride[1]:stride[1] + size[1], stride[2]:stride[2] + size[2]]


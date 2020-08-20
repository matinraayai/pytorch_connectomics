import numpy as np


def make_2d_nhood(radius=1):
    """
    From Janelia pyGreentea: https://github.com/naibaf7/PyGreentea
    Makes neighborhood structures for some mose used dense graphs.
    """

    ceil_rad = np.ceil(radius)
    x = np.arange(-ceil_rad, ceil_rad + 1, 1)
    y = np.arange(-ceil_rad, ceil_rad + 1, 1)
    [i, j] = np.meshgrid(y, x)

    idx_keep = (i**2+j**2) <= radius**2
    i = i[idx_keep].ravel()
    j = j[idx_keep].ravel()
    zero_idx = np.ceil(len(i) / 2).astype(np.int32)

    nhood = np.vstack((i[:zero_idx], j[:zero_idx])).T.astype(np.int32)
    nhood = np.ascontiguousarray(np.flipud(nhood))
    nhood = nhood[1:]
    return nhood


def make_3d_nhood(radius=1):
    """
    Makes nhood structures for some most used dense graphs.
    The neighborhood reference for the dense graph representation we use
    nhood(1,:) is a 3 vector that describe the node that conn(:,:,:,1) connects to
    so to use it: conn(23,12,42,3) is the edge between node [23 12 42] and [23 12 42]+nhood(3,:)
    See? It's simple! nhood is just the offset vector that the edge corresponds to.
    """
    ceil_rad = np.ceil(radius)
    x = np.arange(-ceil_rad, ceil_rad + 1, 1)
    y = np.arange(-ceil_rad, ceil_rad + 1, 1)
    z = np.arange(-ceil_rad, ceil_rad + 1, 1)
    [i, j, k] = np.meshgrid(z, y, x)

    idx_keep = (i**2 + j**2 + k**2) <= radius**2
    i = i[idx_keep].ravel()
    j = j[idx_keep].ravel()
    k = k[idx_keep].ravel()
    zero_idx = np.array(len(i) // 2).astype(np.int32)

    nhood = np.vstack((k[:zero_idx], i[:zero_idx], j[:zero_idx])).T.astype(np.int32)
    return np.ascontiguousarray(np.flipud(nhood))


def make_3d_nhood_aniso(radius_xy=1, radius_xyz_minus1=1.8):
    """
    Makes neighborhood structures for some most used dense graphs.
    """
    nhood_xyz = make_3d_nhood(radius_xy)
    nhood_xyz_minus1 = make_2d_nhood(radius_xyz_minus1)
    nhood = np.zeros((nhood_xyz.shape[0] + 2 * nhood_xyz_minus1.shape[0], 3), dtype=np.int32)
    nhood[:3, :3] = nhood_xyz
    nhood[3:, 0] = -1
    nhood[3:, 1:] = np.vstack((nhood_xyz_minus1, -nhood_xyz_minus1))

    return np.ascontiguousarray(nhood)


def seg_to_aff(seg, nhood=make_3d_nhood(1), pad='replicate'):
    """
    Constructs an affinity graph from a segmentation.
    Assumes affinity graph is represented as:
        shape = (e, z, y, x)
        nhood.shape = (edges, 3)
    """
    shape = seg.shape
    n_edge = nhood.shape[0]
    aff = np.zeros((n_edge,) + shape, dtype=np.float32)

    if len(shape) == 3:  # 3D affinity
        for e in range(n_edge):
            aff[e,
                max(0, -nhood[e, 0]):min(shape[0], shape[0] - nhood[e, 0]),
                max(0, -nhood[e, 1]):min(shape[1], shape[1] - nhood[e, 1]),
                max(0, -nhood[e, 2]):min(shape[2], shape[2] - nhood[e, 2])] = \
                (seg[max(0, -nhood[e, 0]):min(shape[0], shape[0] - nhood[e, 0]),
                     max(0, -nhood[e, 1]):min(shape[1], shape[1] - nhood[e, 1]),
                     max(0, -nhood[e, 2]):min(shape[2], shape[2] - nhood[e, 2])] ==
                 seg[max(0, nhood[e, 0]):min(shape[0], shape[0] + nhood[e, 0]),
                     max(0, nhood[e, 1]):min(shape[1], shape[1] + nhood[e, 1]),
                     max(0, nhood[e, 2]):min(shape[2], shape[2] + nhood[e, 2])]) \
                 * ( seg[max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
                                max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1]), \
                                max(0,-nhood[e,2]):min(shape[2],shape[2]-nhood[e,2])] > 0 ) \
                            * ( seg[max(0,nhood[e,0]):min(shape[0],shape[0]+nhood[e,0]), \
                                max(0,nhood[e,1]):min(shape[1],shape[1]+nhood[e,1]), \
                                max(0,nhood[e,2]):min(shape[2],shape[2]+nhood[e,2])] > 0 )
    elif len(shape) == 2: # 2D affinity
        for e in range(n_edge):
            aff[e,
                max(0, -nhood[e, 0]):min(shape[0], shape[0] - nhood[e, 0]),
                max(0, -nhood[e, 1]):min(shape[1], shape[1] - nhood[e, 1])] = \
                (seg[max(0, -nhood[e, 0]):min(shape[0], shape[0] - nhood[e, 0]),
                     max(0, -nhood[e, 1]):min(shape[1], shape[1] - nhood[e, 1])] ==
                 seg[max(0, nhood[e, 0]):min(shape[0], shape[0] + nhood[e, 0]),
                     max(0, nhood[e, 1]):min(shape[1], shape[1] + nhood[e, 1])]) \
                 * (seg[max(0, -nhood[e, 0]):min(shape[0], shape[0] - nhood[e, 0]), \
                                max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1])] > 0 ) \
                            * ( seg[max(0,nhood[e,0]):min(shape[0],shape[0]+nhood[e,0]), \
                                max(0,nhood[e,1]):min(shape[1],shape[1]+nhood[e,1])] > 0 )

    if n_edge == 3 and pad == 'replicate':  # pad the boundary affinity
        aff[0, 0] = (seg[0] > 0).astype(aff.dtype)
        aff[1, :, 0] = (seg[:, 0] > 0).astype(aff.dtype)
        aff[2, :, :, 0] = (seg[:, :, 0] > 0).astype(aff.dtype)
    elif n_edge == 2 and pad == 'replicate':  # pad the boundary affinity
        aff[0, 0] = (seg[0] > 0).astype(aff.dtype)
        aff[1, :, 0] = (seg[:, 0] > 0).astype(aff.dtype)

    return aff

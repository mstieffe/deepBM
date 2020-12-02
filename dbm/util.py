import math
from functools import reduce

import numpy as np
import torch
from torch.nn.modules.module import _addindent
import os

class Stats(object):
    def __init__(self, group, items):
        self._stats = []
        self._group = group
        self._items = items

    def add(self, tup):
        if len(tup) != len(self._items):
            raise ValueError("Incompatible stats")
        self._stats.append(torch.stack([x.detach() for x in tup]))

    def samples(self):
        return np.array(self._stats)

    def write(self, writer, step, clear: bool = True):
        stats = torch.stack(self._stats)
        for ix, item in enumerate(self._items):
            name = self._group + "/" + item
            value = torch.mean(stats[..., ix]).cpu().numpy()
            writer.add_scalar(name, value, global_step=step)
        if clear:
            self.clear()

    def clear(self):
        self._stats.clear()


def compute_same_padding(kernel_size, stride, dilation):
    if kernel_size % 2 == 0:
        raise ValueError("Only wörks for odd kernel sizes.")
    out = math.ceil((1 - stride + dilation * (kernel_size - 1)) / 2)
    return max(out, 0)


def calculate_output_dim(input_dim, kernel_size, stride, dilation, padding):
    return math.floor(
        (input_dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    )


def voxel_gauss(coords, res=64, width=200, sigma=100.0, device=None):
    grid = torch.stack(make_grid(res=res, width=width, device=device))
    coords = coords.view(*coords.shape, 1, 1)
    grid = torch.exp(-torch.div(torch.sum((grid - coords) ** 2, dim=2), sigma))
    return grid


def voxelize_gauss(coord_inp, sigma, grid):
    coords = coord_inp[..., None, None, None]
    voxels = np.exp(-1.0 * np.sum((grid - coords) * (grid - coords), axis=2) / sigma)
    # voxels = np.transpose(voxels, [0, 2, 3, 4, 1])
    return voxels.squeeze().astype(np.float32)


def voxelize_gauss_batch(coord_inp, sigma, grid):
    coords = coord_inp[..., None, None, None]
    grid = grid[None, ...]
    voxels = np.exp(-1.0 * np.sum((grid - coords) * (grid - coords), axis=3) / sigma)
    # voxels = np.transpose(voxels, [0, 2, 3, 4, 1])
    return voxels.squeeze().astype(np.float32)

def make_grid_np(ds, grid_size):
    grid = np.arange(-int(grid_size / 2), int(grid_size / 2), 1.0).astype(np.float32)
    grid += 0.5
    grid *= ds
    X, Y, Z = np.meshgrid(grid, grid, grid, indexing="ij")
    grid = np.stack([X, Y, Z])[None, None, ...]
    return grid


def make_grid(res, width, device=None):
    grid = (width / res) * (
        torch.arange(-int(res / 2), int(res / 2), device=device, dtype=torch.float)
        + 0.5
    )
    return torch.meshgrid(grid, grid, grid)


def avg_blob(grid, res=64, width=200, sigma=100.0, device=None):
    X, Y, Z = make_grid(res, width, device=device)
    X = X.view(1, 1, *X.shape)
    Y = Y.view(1, 1, *Y.shape)
    Z = Z.view(1, 1, *Z.shape)
    reduction_dims = (-1, -2, -3)
    grid = grid / torch.sum(grid, dim=reduction_dims, keepdim=True)
    X = torch.sum(grid * X, dim=reduction_dims)
    Y = torch.sum(grid * Y, dim=reduction_dims)
    Z = torch.sum(grid * Z, dim=reduction_dims)
    coords = torch.stack((X, Y, Z), dim=2)
    return coords


def rand_rotation_matrix(deflection=1.0, randnums=None):
    if randnums is None:
        randnums = np.random.uniform(size=(3,))

    theta, phi, z = randnums

    theta = theta * 2.0 * deflection * np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0 * np.pi  # For direction of pole deflection.
    z = z * 2.0 * deflection  # For magnitude of pole deflection.

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.

    r = np.sqrt(z)
    V = (np.sin(phi) * r, np.cos(phi) * r, np.sqrt(2.0 - z))

    st = np.sin(theta)
    ct = np.cos(theta)

    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

    # Construct the rotation matrix  ( V Transpose(V) - I ) R.

    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M


def summary(model, log):
    """ Copied and modified from https://github.com/pytorch/pytorch/issues/2001 """
    def repr(model):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = model.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        total_params = 0
        for key, module in model._modules.items():
            mod_str, num_params = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
            total_params += num_params
        lines = extra_lines + child_lines

        for name, p in model._parameters.items():
            total_params += reduce(lambda x, y: x * y, p.shape)

        main_str = model._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        main_str += ', \033[92m{:,}\033[0m params'.format(total_params)
        return main_str, total_params

    string, count = repr(model)
    log.warning(string)
    log.warning("Total # parameters: %s", count)
    return count

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
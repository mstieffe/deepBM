import tensorflow as tf
import numpy as np
#from tqdm import tqdm


# Map coords to grid
@tf.function
def voxelize_gauss(coord_inp, sigma, grid_size, ds):
    grid = tf.range(- int(grid_size / 2), int(grid_size / 2), dtype=tf.float32)
    grid = tf.add(grid, 0.5)
    grid = tf.scalar_mul(ds, grid)

    X, Y, Z = tf.meshgrid(grid, grid, grid, indexing='ij')
    grid = tf.stack([X, Y, Z])
    grid = tf.expand_dims(grid, 0)
    grid = tf.expand_dims(grid, 0)

    coords = tf.expand_dims(coord_inp, axis=-1)
    coords = tf.expand_dims(coords, axis=-1)
    coords = tf.expand_dims(coords, axis=-1)

    coords = tf.cast(coords, tf.float32)
    grid = tf.subtract(grid, coords)
    grid = tf.square(grid)
    grid = tf.reduce_sum(grid, axis=2)
    grid = tf.divide(grid, sigma)
    grid = tf.scalar_mul(-1.0, grid)
    grid = tf.exp(grid)
    grid = tf.transpose(grid, [0, 2, 3, 4, 1])
    return grid

@tf.function
def prepare_initial(atom_pos, bead_pos, bead_featvec, sigma, grid_size, ds):
    bead_featvec = tf.cast(bead_featvec, tf.float32)
    bead_featvec = bead_featvec[:, tf.newaxis, tf.newaxis, tf.newaxis, :, :]

    atom_grid = voxelize_gauss(atom_pos, sigma, grid_size, ds)
    #atom_grid = atom_grid[:, :, :, :, :, tf.newaxis]
    #atom_grid = tf.reduce_sum(atom_grid * atom_featvec, axis = 4)

    bead_grid = voxelize_gauss(bead_pos, sigma, grid_size, ds)
    bead_grid = bead_grid[:, :, :, :, :, tf.newaxis]
    bead_grid = tf.reduce_sum(bead_grid * bead_featvec, axis = 4)

    #env_grid = atom_grid + bead_grid
    return atom_grid, atom_grid, bead_grid

@tf.function
def prepare_initial_predict(atom_pos, bead_pos, bead_featvec, sigma, grid_size, ds):
    bead_featvec = tf.cast(bead_featvec, tf.float32)
    bead_featvec = bead_featvec[:, tf.newaxis, tf.newaxis, tf.newaxis, :, :]

    atom_grid = voxelize_gauss(atom_pos, sigma, grid_size, ds)
    #atom_grid = atom_grid[:, :, :, :, :, tf.newaxis]
    #atom_grid = tf.reduce_sum(atom_grid * atom_featvec, axis = 4)

    bead_grid = voxelize_gauss(bead_pos, sigma, grid_size, ds)
    bead_grid = bead_grid[:, :, :, :, :, tf.newaxis]
    bead_grid = tf.reduce_sum(bead_grid * bead_featvec, axis = 4)

    #env_grid = atom_grid + bead_grid
    return atom_grid, bead_grid

@tf.function
def prepare_grid(atom_pos, atom_featvec, bead_pos, bead_featvec, sigma, grid_size, ds):

    atom_featvec = tf.cast(atom_featvec, tf.float32)
    atom_featvec = atom_featvec[:, tf.newaxis, tf.newaxis, tf.newaxis, :, :]
    bead_featvec = tf.cast(bead_featvec, tf.float32)
    bead_featvec = bead_featvec[:, tf.newaxis, tf.newaxis, tf.newaxis, :, :]

    atom_grid = voxelize_gauss(atom_pos, sigma, grid_size, ds)
    atom_grid = atom_grid[:, :, :, :, :, tf.newaxis]
    atom_grid = tf.reduce_sum(atom_grid * atom_featvec, axis = 4)

    bead_grid = voxelize_gauss(bead_pos, sigma, grid_size, ds)
    bead_grid = bead_grid[:, :, :, :, :, tf.newaxis]
    bead_grid = tf.reduce_sum(bead_grid * bead_featvec, axis = 4)

    env_grid = atom_grid + bead_grid
    return env_grid

@tf.function
def prepare_target(target_pos, target_featvec, sigma, grid_size, ds):
    target_featvec = tf.cast(target_featvec, tf.float32)
    target_featvec = target_featvec[:, tf.newaxis, tf.newaxis, :, :]

    target_grid = voxelize_gauss(target_pos, sigma, grid_size, ds)
    target_grid = target_grid * target_featvec

    return target_grid
    
@tf.function
def average_blob_pos(grid, grid_size, ds):
    g = tf.range(- int(grid_size / 2), int(grid_size / 2), dtype=tf.float32)
    g = tf.add(g, 0.5)
    g = tf.scalar_mul(ds, g)
    X, Y, Z = tf.meshgrid(g, g, g, indexing='ij')

    X = tf.expand_dims(X, 0)
    X = tf.expand_dims(X, -1)

    Y = tf.expand_dims(Y, 0)
    Y = tf.expand_dims(Y, -1)

    Z = tf.expand_dims(Z, 0)
    Z = tf.expand_dims(Z, -1)

    grid = tf.cast(grid, tf.float32)
    grid_sum = tf.reduce_sum(grid, axis=[1, 2, 3], keepdims=True)
    grid_sum = tf.add(grid_sum, 1E-20)

    grid = tf.divide(grid, grid_sum)

    X = tf.multiply(grid, X)
    X = tf.reduce_sum(X, axis=[1, 2, 3])

    Y = tf.multiply(grid, Y)
    Y = tf.reduce_sum(Y, axis=[1, 2, 3])

    Z = tf.multiply(grid, Z)
    Z = tf.reduce_sum(Z, axis=[1, 2, 3])

    Coords = tf.stack([X, Y, Z], axis=2)

    return Coords



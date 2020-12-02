import torch
from dbm.ff import *
import sys


def bond_energy(atoms, indices, params):
    ndx1 = indices[:, :, 1] # (BS, n_bonds)
    ndx2 = indices[:, :, 2]
    param_ndx = indices[:, :, 0]

    param = torch.gather(params, 0, param_ndx)
    a_0 = param[:, :, 0]
    f_c = param[:, :, 1]

    pos1 = torch.gather(atoms, 1, ndx1) # (BS, n_bonds, 3)
    pos2 = torch.gather(atoms, 1, ndx2)

    #tf.print(f_c, output_stream=sys.stdout)

    dis = pos1 - pos2
    dis = dis**2
    dis = torch.sum(dis, 2)
    dis = torch.sqrt(dis)

    #dis = tf.clip_by_value(dis, 10E-8, 1000.0)
    dis = torch.where(dis > 0.01, dis, torch.tensor(0.01))


    en = dis - a_0
    en = en**2

    en = en * f_c / 2.0

    en = torch.sum(en, 1)

    return en

@tf.function
def angle_energy(atoms, indices, params):
    ndx1 = indices[:, :, 1] # (BS, n_angles)
    ndx2 = indices[:, :, 2]
    ndx3 = indices[:, :, 3]
    param_ndx = indices[:, :, 0]

    param = tf.gather(params, param_ndx, axis=0)
    a_0 = param[:, :, 0]
    f_c = param[:, :, 1]

    #tf.print(a_0, output_stream=sys.stdout)


    pos1 = tf.gather(atoms, ndx1, batch_dims=1, axis=1) # (BS, n_angles, 3)
    pos2 = tf.gather(atoms, ndx2, batch_dims=1, axis=1)
    pos3 = tf.gather(atoms, ndx3, batch_dims=1, axis=1)

    #tf.print(pos1, summarize=-1)
    #tf.print(pos2, summarize=-1)
    #tf.print(pos3, summarize=-1)

    vec1 = tf.subtract(pos1, pos2)
    vec2 = tf.subtract(pos3, pos2)

    norm1 = tf.square(vec1)
    norm1 = tf.reduce_sum(norm1, axis=2)
    norm1 = tf.sqrt(norm1)
    norm2 = tf.square(vec2)
    norm2 = tf.reduce_sum(norm2, axis=2)
    norm2 = tf.sqrt(norm2)
    norm = tf.multiply(norm1, norm2)

    dot = tf.multiply(vec1, vec2)
    dot = tf.reduce_sum(dot, axis=2)

    #norm = tf.clip_by_value(norm, 10E-8, 1000.0)

    a = tf.divide(dot, norm)
    a = tf.clip_by_value(a, -0.999, 0.999)
    #a = tf.clip_by_value(a, -0.9999, 0.9999)  # prevent nan because of rounding errors

    # tf.acos should return angle in radiant??
    a = tf.acos(a)
    #tf.print(a)
    #tf.print(a, output_stream=sys.stdout)

    en = tf.subtract(a, a_0)
    en = tf.square(en)
    en = tf.multiply(en, f_c)
    en = tf.divide(en, 2.0)
    en = tf.reduce_sum(en, axis=1)
    return en


def dih_energy(atoms, indices, params):
    ndx1 = indices[:, :, 1] # (BS, n_dihs)
    ndx2 = indices[:, :, 2]
    ndx3 = indices[:, :, 3]
    ndx4 = indices[:, :, 4]
    param_ndx = indices[:, :, 0]

    param = tf.gather(params, param_ndx, axis=0)
    a_0 = param[:, :, 0]
    f_c = param[:, :, 1]
    func_type = tf.cast(param[:, :, 2], tf.int32)
    mult = param[:, :, 3]

    #tf.print(a_0, output_stream=sys.stdout)

    pos1 = tf.gather(atoms, ndx1, batch_dims=1, axis=1) # (BS, n_dihs, 3)
    pos2 = tf.gather(atoms, ndx2, batch_dims=1, axis=1)
    pos3 = tf.gather(atoms, ndx3, batch_dims=1, axis=1)
    pos4 = tf.gather(atoms, ndx4, batch_dims=1, axis=1)

    vec1 = tf.subtract(pos2, pos1)
    vec2 = tf.subtract(pos2, pos3)
    vec3 = tf.subtract(pos4, pos3)

    plane1 = tf.linalg.cross(vec1, vec2)
    plane2 = tf.linalg.cross(vec2, vec3)

    norm1 = tf.square(plane1)
    norm1 = tf.reduce_sum(norm1, axis=2)
    norm1 = tf.sqrt(norm1)

    norm2 = tf.square(plane2)
    norm2 = tf.reduce_sum(norm2, axis=2)
    norm2 = tf.sqrt(norm2)

    dot = tf.multiply(plane1, plane2)
    dot = tf.reduce_sum(dot, axis=2)

    norm = tf.multiply(norm1, norm2) #+ 1E-20
    a = tf.divide(dot, norm)
    a = tf.clip_by_value(a, -0.999, 0.999)



    #a = tf.clip_by_value(a, -0.9999, 0.9999)  # prevent nan because of rounding errors

    a = tf.acos(a)
    #tf.print(a, output_stream=sys.stdout)

    a = tf.where(func_type == 1, tf.multiply(a, 3.0), a)
    #a = tf.multiply(a, 3.0)
    #tf.print(func_type, output_stream=sys.stdout, summarize=-1)
    #tf.print(a_0, output_stream=sys.stdout, summarize=-1)
    #tf.print(a, output_stream=sys.stdout, summarize=-1)

    en = tf.subtract(a, a_0)

    en = tf.where(func_type == 1, tf.multiply(tf.add(tf.cos(en), 1.0), f_c), tf.multiply(tf.square(en), f_c / 2.0))
    #en = tf.where(func_type == 1, tf.multiply(tf.add(tf.cos(en), 1.0), f_c), 0.0)
    #en = tf.where(func_type == 1, 0.0, tf.multiply(tf.square(en), f_c / 2.0))
    #tf.print(en, output_stream=sys.stdout, summarize=-1)

    #tf.print(en, output_stream=sys.stdout, summarize=-1)


    #en = tf.cos(en)
    #en = tf.add(en, 1.0)
    #en = tf.multiply(en, f_c)
    en = tf.reduce_sum(en, axis=1)
    return en

@tf.function
def lj_energy(atoms, indices, params):
    #tf.print(indices, output_stream=sys.stdout, summarize=-1)

    ndx1 = indices[:, :, 1] # (BS, n_ljs)
    ndx2 = indices[:, :, 2]
    param_ndx = indices[:, :, 0]

    param = tf.gather(params, param_ndx, axis=0)
    sigma = param[:, :, 0]
    epsilon = param[:, :, 1]

    pos1 = tf.gather(atoms, ndx1, batch_dims=1, axis=1) # (BS, n_ljs, 3)
    pos2 = tf.gather(atoms, ndx2, batch_dims=1, axis=1)

    dis = tf.subtract(pos1, pos2)
    dis = tf.square(dis)
    dis = tf.reduce_sum(dis, axis=2)
    dis = tf.sqrt(dis)

    dis = tf.maximum(dis, 0.001)

    #tf.print(dis, output_stream=sys.stdout, summarize=-1)

    #tf.print(tf.math.reduce_min(dis), output_stream=sys.stdout, summarize=-1)
    #test = tf.where(sigma != 0.0, dis, 100.0)
    #tf.print(tf.math.reduce_min(test), output_stream=sys.stdout, summarize=-1)

    #r_6 = tf.pow(dis, 6)
    #r_12 = tf.pow(r_6, 2)

    c6_term = tf.divide(sigma, dis)
    c6_term = tf.pow(c6_term, 6)
    c12_term = tf.pow(c6_term, 2)

    #tf.print(c6_term, output_stream=sys.stdout, summarize=-1)
    #tf.print(c12_term, output_stream=sys.stdout, summarize=-1)

    #tf.print(c6_term, output_stream=sys.stdout, summarize=-1)
    #tf.print(c12_term, output_stream=sys.stdout, summarize=-1)
    #tf.print(epsilon, output_stream=sys.stdout, summarize=-1)
    #print("neee")

    en = tf.subtract(c12_term, c6_term)
    en = tf.multiply(en, 4*epsilon)
    #tf.print(en, output_stream=sys.stdout, summarize=-1)
    #tf.print(en, output_stream=sys.stdout, summarize=-1)

    """
    c6_term_cut = tf.divide(sigma, 1.0)
    c6_term_cut = tf.pow(c6_term_cut, 6)
    c12_term_cut = tf.pow(c6_term_cut, 2)
    en_cut = tf.subtract(c12_term_cut, c6_term_cut)
    en_cut = tf.multiply(en_cut, 4*epsilon)
    en = en - en_cut
    en = tf.where(dis <= 1.0, en, 0.0)

    """
    #tf.print(tf.math.reduce_max(en), output_stream=sys.stdout, summarize=-1)


    en = tf.reduce_sum(en, axis=1)

    return en

#print(bond_params)
#print(angle_params)
#print(dih_params)
#print(lj_params)
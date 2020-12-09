import numpy as np
import os
from os import path
import pickle
import mdtraj as md
import networkx as nx
# from tqdm.auto import tqdm
from timeit import default_timer as timer
import itertools
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from dbm.ff import *
from dbm.universe1 import *
#from dbm.env import *
#from utils import *
from copy import deepcopy
from operator import add


class Data():

    def __init__(self, folders, ff_file, cutoff=0.7, kick=0.03, align=False, aug=False, order="dfs", heavy_first=False, cg_dropout=0.0, save=True):


        #load or create FF
        ff_name = os.path.splitext(ff_file.split("/")[-1])[0]
        if path.exists("./forcefield/processed/"+ff_name):
            with open("./forcefield/processed/"+ff_name, 'rb') as input:
                self.ff = pickle.load(input)
            print("loaded FF")
        else:
            self.make_dir("./forcefield/processed/")
            self.ff = FF(ff_file)
            if save:
                with open("./forcefield/processed/"+ff_name, 'wb') as output:
                    pickle.dump(self.ff, output, pickle.HIGHEST_PROTOCOL)

        self.align = align
        self.aug = aug
        self.order = order
        self.heavy_first = heavy_first
        self.cg_dropout = cg_dropout

        self.samples = []

        start_setup = timer()
        print("Setting up universe. This may take a while...")

        #processed_folder = "./data/processed/"
        #self.make_dir(processed_folder)
        #processed_name = '-'.join(folders) + '_cutoff={}_kick={}_align={}_aug={}_fix_seq={}.pkl'.format(cutoff, kick, align, aug, fix_seq)
        print((cutoff, kick, align, aug))

        self.folder_dict = {}


        for folder in folders:
            samples = []

            self.make_dir(folder + "/processed")
            tag = '-'.join([os.path.splitext(t)[0] for t in os.listdir(folder + "/cg")])
            tag = tag + '_cutoff={}_kick={}_align={}_aug={}_ord={}_hf={}_ff={}_drop={}.pkl'.format(cutoff, kick, align, aug, order, heavy_first, self.ff.name, cg_dropout)
            processed_file = folder + "/processed/" + tag

            if path.exists(processed_file):
                start = timer()
                with open(processed_file, 'rb') as input:
                    samples = pickle.load(input)
                print("Loaded train universe from " + processed_file + 'it took ', timer() - start, 'secs')
            else:
                #self.make_dir(folder + '/processed')
                for file in os.listdir(folder + "/cg"):
                    if file.endswith(".gro"):
                        start = timer()
                        cg_file = os.path.join(folder + "/cg", file)
                        print("processing '", cg_file, "'", end='')
                        aa_file = None
                        if path.exists(os.path.join(folder + "/aa", file)):
                            aa_file = os.path.join(folder + "/aa", file)
                            print(" with reference data '", aa_file, "'", end='')
                        print(" ... ", end='')
                        u = Universe(self.ff, folder, cg_file, aa_file, cutoff=cutoff, kick=kick, align=align, aug=aug, order=order, heavy_first=heavy_first, cg_dropout=cg_dropout)
                        print("done! it took ", timer() - start, "secs")
                        samples.append(u)
                if save:
                    with open(processed_file, 'wb') as output:
                        pickle.dump(samples, output, pickle.HIGHEST_PROTOCOL)

            self.folder_dict[os.path.splitext(folder.split("/")[-1])[0]] = samples
            self.samples += samples

        #sanity check
        for u in self.samples:
            if u.ff.file != self.ff.file:
                raise Exception('Forcefields do not match')

        end_setup = timer()
        print("Successfully created universe! This took ", end_setup-start_setup, "secs")

        #find maximums for padding
        self.update_max_values()

        self.num_u = len(self.samples)

    def __call__(self):
        return self.recurrent_generator_combined(train=True)

    def update_max_values(self):
        max_seq_len = max([u.max_seq_len for u in self.samples])
        max_atoms = max([u.max_atoms for u in self.samples])
        max_beads = max([u.max_beads for u in self.samples])
        max_bonds_pb = max([u.max_bonds_pb for u in self.samples])
        max_angles_pb = max([u.max_angles_pb for u in self.samples])
        max_dihs_pb = max([u.max_dihs_pb for u in self.samples])
        max_ljs_pb = max([u.max_ljs_pb for u in self.samples])
        max_bonds = max([u.max_bonds for u in self.samples])
        max_angles = max([u.max_angles for u in self.samples])
        max_dihs = max([u.max_dihs for u in self.samples])
        max_ljs = max([u.max_ljs for u in self.samples])

        for u in self.samples:
            u.max_seq_len = max_seq_len
            u.max_atoms = max_atoms
            u.max_beads = max_beads
            u.max_bonds_pb = max_bonds_pb
            u.max_angles_pb = max_angles_pb
            u.max_dihs_pb = max_dihs_pb
            u.max_ljs_pb = max_ljs_pb
            u.max_bonds = max_bonds
            u.max_angles = max_angles
            u.max_dihs = max_dihs
            u.max_ljs = max_ljs

    def make_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def kick_atoms(self):
        for u in self.samples:
            u.kick_atoms()

    def update_pos2(self, atoms, new_coords, bm_mode="normal", temp=300):
        if bm_mode == "mc":
            red_temp = 1.66054E-21/(1.380649E-23*temp)
        for u, a, coord in zip(self.samples, atoms, new_coords):
            #a = f.atom
            if bm_mode == "mc":
                energy_old = f[a].energy()
                pos_old = a.pos
                a.pos = f.rot_back(coord)
                energy_new = f.energy()
                delta_e = energy_new - energy_old
                if delta_e > 0.0:
                    if np.random.uniform() > np.exp(-delta_e*red_temp):
                        a.pos = pos_old
            if bm_mode == "steep":
                energy_old = f.energy()
                pos_old = a.pos
                a.pos = f.rot_back(coord)
                energy_new = f.energy()
                delta_e = energy_new - energy_old
                if delta_e > 0.0:
                    a.pos = pos_old
            else:
                a.pos = u.features[a].rot_back(coord)

    def update_pos(self, features, new_coords, bm_mode="normal", temp=300):
        if bm_mode == "mc":
            red_temp = 1.66054E-21/(1.380649E-23*temp)
        for f, coord in zip(features, new_coords):
            a = f.atom
            if bm_mode == "mc":
                energy_old = f[a].energy()
                pos_old = a.pos
                a.pos = f.rot_back(coord)
                energy_new = f.energy()
                delta_e = energy_new - energy_old
                if delta_e > 0.0:
                    if np.random.uniform() > np.exp(-delta_e*red_temp):
                        a.pos = pos_old
            if bm_mode == "steep":
                energy_old = f.energy()
                pos_old = a.pos
                a.pos = f.rot_back(coord)
                energy_new = f.energy()
                delta_e = energy_new - energy_old
                if delta_e > 0.0:
                    a.pos = pos_old
            else:
                a.pos = f.rot_back(coord)


    def make_batch_old(self, mode="init"):
        iters = []
        for u in self.samples:
            iters.append(u.generator(train=False, mode=mode))
        bs = self.num_u
        batch = []
        for generators in zip(*iters):
            for gen_elem in generators:
                batch.append(gen_elem)
                if len(batch) == bs:
                    yield zip(*batch)
                    batch = []

    def make_recurrent_batch(self, bs=16, train=False, mode='init', cg_kick=0.0):
        #m = 0
        for u in self.samples:
            if self.heavy_first:
                generator = u.recurrent_generator_heavyfirst(train=train, mode=mode, rand_rot=False, cg_kick=cg_kick)
            else:
                generator = u.recurrent_generator(train=train, mode=mode, rand_rot=False, cg_kick=cg_kick)
            for elem in generator:
                #print(m)
                #m = m+1
                atoms, _, target_type, atom_featvec, repl, _,  a_pos, b_pos, bead_featvec, b_ndx, a_ndx, d_ndx, lj_ndx = elem
                atom_pos, bead_pos = [], []
                for n in range(0, bs):
                    rot_mat = self.rot_mat_z(np.pi*2*n/bs)
                    atom_pos.append(np.dot(a_pos, rot_mat))
                    bead_pos.append(np.dot(b_pos, rot_mat))
                    #atom_pos.append(a_pos)
                    #bead_pos.append(b_pos)
                atom_pos = np.array(atom_pos)
                bead_pos = np.array(bead_pos)
                bead_featvec = np.array([bead_featvec] * bs)

                target_type = np.transpose(np.array([target_type]*bs), [1,0,2])
                atom_featvec = np.transpose(np.array([atom_featvec]*bs), [1,0,2,3])
                repl = np.transpose(np.array([repl]*bs), [1, 0, 2])

                b_ndx = np.array([b_ndx]*bs)
                a_ndx = np.array([a_ndx]*bs)
                d_ndx = np.array([d_ndx]*bs)
                lj_ndx = np.array([lj_ndx]*bs)

                features = []
                for a in atoms:
                    features.append(u.features[a])
                yield features, target_type, atom_featvec, repl, atom_pos, bead_pos, bead_featvec, b_ndx, a_ndx, d_ndx, lj_ndx

    def make_recurrent_batch2(self, bs=16, train=False, mode='init', cg_kick=0.0):
        #m = 0
        for u in self.samples:
            if self.heavy_first:
                generator = u.recurrent_generator_heavyfirst2(train=train, mode=mode, rand_rot=False, cg_kick=cg_kick)
            else:
                generator = u.recurrent_generator(train=train, mode=mode, rand_rot=False, cg_kick=cg_kick)
            for elem in generator:
                #print(m)
                #m = m+1
                atoms, _, target_type, atom_featvec, repl, _,  a_pos, b_pos, bead_featvec, b_ndx2, a_ndx2, d_ndx2, lj_ndx2, b_ndx, a_ndx, d_ndx, lj_ndx = elem
                atom_pos, bead_pos = [], []
                for n in range(0, bs):
                    rot_mat = self.rot_mat_z(np.pi*2*n/bs)
                    atom_pos.append(np.dot(a_pos, rot_mat))
                    bead_pos.append(np.dot(b_pos, rot_mat))
                    #atom_pos.append(a_pos)
                    #bead_pos.append(b_pos)
                atom_pos = np.array(atom_pos)
                bead_pos = np.array(bead_pos)
                bead_featvec = np.array([bead_featvec] * bs)


                b_ndx = np.array([b_ndx]*bs)
                a_ndx = np.array([a_ndx]*bs)
                d_ndx = np.array([d_ndx]*bs)
                lj_ndx = np.array([lj_ndx]*bs)
                b_ndx2 = np.transpose(np.array([b_ndx2]*bs), [1,0,2, 3])
                a_ndx2 = np.transpose(np.array([a_ndx2]*bs), [1,0,2, 3])
                d_ndx2 = np.transpose(np.array([d_ndx2]*bs), [1,0,2, 3])
                lj_ndx2 = np.transpose(np.array([lj_ndx2]*bs), [1,0,2, 3])

                target_type = np.transpose(np.array([target_type]*bs), [1,0,2])
                atom_featvec = np.transpose(np.array([atom_featvec]*bs), [1,0,2,3])
                repl = np.transpose(np.array([repl]*bs), [1, 0, 2])


                features = []
                for a in atoms:
                    features.append(u.features[a])
                yield features, target_type, atom_featvec, repl, atom_pos, bead_pos, bead_featvec, b_ndx, a_ndx, d_ndx, lj_ndx, b_ndx2, a_ndx2, d_ndx2, lj_ndx2

    def rot_mat_z(self, theta):
        #rotation axis
        v_rot = np.array([0.0, 0.0, 1.0])

        #rotation angle
        #theta = np.random.uniform(0, np.pi * 2)

        #rotation matrix
        a = math.cos(theta / 2.0)
        b, c, d = -v_rot * math.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        rot_mat = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                         [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                         [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

        return rot_mat

    def make_batch(self, mode="init"):
        c = 0
        iters = []
        for u in self.samples:
            if self.heavy_first:
                iters.append(u.generator_heavyfirst(train=False, mode=mode))
            else:
                iters.append(u.generator(train=False, mode=mode))
        bs = self.num_u
        batch = []

        while True:
            if bs == 0:
                break
            for iter in iters:
                try:
                    gen_elem = next(iter)
                    batch.append(gen_elem)
                except StopIteration:
                    iters.remove(iter)
                    bs = bs -1
                if len(batch) == bs and bs != 0:
                    c = c+1
                    #print(c, len(batch))
                    yield zip(*batch)
                    batch = []

    def make_set(self, train=True, cg_kick=0.0):
        ds = []
        for u in self.samples:
            ds += list(u.recurrent_generator_combined(train=train, rand_rot=False, cg_kick=cg_kick))
        return ds

    def make_set2(self, train=True, cg_kick=0.0):
        ds = []
        for u in self.samples:
            ds += list(u.recurrent_generator_combined2(train=train, rand_rot=False, cg_kick=cg_kick))
        return ds

    def recurrent_generator_combined(self, train=True, rand_rot=False, cg_kick=0.0):
        iters = []
        for u in self.samples:
            iters.append(u.recurrent_generator_combined(train=train, rand_rot=rand_rot, cg_kick=cg_kick))
        for generators in zip(*iters):
            for gen_elem in generators:
                yield gen_elem

    def recurrent_generator(self, train=True, mode='init', rand_rot=False, cg_kick=0.0):
        iters = []
        for u in self.samples:
            if self.heavy_first:
                iters.append(u.recurrent_generator_heavyfirst(train=train, mode=mode, rand_rot=rand_rot, cg_kick=cg_kick))
            else:
                iters.append(u.recurrent_generator(train=train, mode=mode, rand_rot=rand_rot, cg_kick=cg_kick))
        for generators in zip(*iters):
            for gen_elem in generators:
                yield gen_elem

    def generator_combined(self, train=True, rand_rot=False, cg_kick=0.0):
        iters = []
        for u in self.samples:
            iters.append(u.generator_combined(train=train, rand_rot=rand_rot, cg_kick=cg_kick))
        for generators in zip(*iters):
            for gen_elem in generators:
                yield gen_elem

    def generator(self, train=True, mode='init', rand_rot=False, cg_kick=0.0):
        iters = []
        for u in self.samples:
            iters.append(u.generator(train=train, mode=mode, rand_rot=rand_rot, cg_kick=cg_kick))
        for generators in zip(*iters):
            for gen_elem in generators:
                yield gen_elem



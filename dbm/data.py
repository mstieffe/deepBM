import numpy as np
from scipy.stats import entropy
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

    def jsd(self, p, q, base=2.0):
        '''
            Implementation of pairwise `jsd` based on
            https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
        '''
        ## convert to np.array
        p, q = np.asarray(p), np.asarray(q)
        ## normalize p, q to probabilities
        p, q = p / p.sum(), q / q.sum()
        m = 1. / 2 * (p + q)
        return entropy(p, m, base=base) / 2. + entropy(q, m, base=base) / 2.

    def evaluate(self, folder="", tag="", ref=False):
        self.make_dir(folder)
        self.make_dir(folder+"/data")
        data_folder = folder+"/data/"
        for name, samples in zip(self.folder_dict.keys(), self.folder_dict.values()):
            ref_exists = np.all([False for s in samples if s.aa_file == None])
            #LJ

            """
            folder_bead_lj = folder + "/bead_lj/"
            self.make_dir(folder_bead_lj)
            n_beads = len(samples[0].mols[0].beads)
            for n in range(0, n_beads):
                fig = plt.figure(figsize=(12, 12))
                ax = fig.add_subplot(1, 1, 1)
                ax.set_title("LJ", fontsize=4)
                if ref:
                    lj_dstr = samples[0].lj_distribution_bead(n, ref=True)
                    values = list(lj_dstr.values())
                    keys = list(lj_dstr.keys())
                    if len(self.samples) > 1:
                        for u in samples[1:]:
                            lj_dstr = u.lj_distribution_bead(n, ref=True)
                            values = [v1 + v2 for (v1, v2) in zip(values, lj_dstr.values())]
                    if len(values) == len(keys):
                        ax.plot(keys, values, label="ref", color="black", linewidth=2, linestyle='-')

                lj_dstr = samples[0].lj_distribution_bead(n)
                values = list(lj_dstr.values())
                keys = list(lj_dstr.keys())
                if len(samples) > 1:
                    for u in samples[1:]:
                        lj_dstr = u.lj_distribution_bead(n)
                        values = [v1+v2 for (v1,v2) in zip(values,lj_dstr.values())]
                if len(values) == len(keys):
                    ax.plot(keys, values, label="bm", color="red", linewidth=2, linestyle='-')

                ax.set_xlabel('E [kJ/mol]')
                ax.set_ylabel('p')
                plt.legend()
                plt.tight_layout()
                plt.savefig(folder_bead_lj+"lj_"+name+tag+"_"+str(n)+".pdf")
            """

            tot_score = 0.0

            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(1, 1, 1)
            ax.set_title("LJ", fontsize=4)

            lj_dstr = samples[0].lj_distribution()
            values = list(lj_dstr.values())
            keys = list(lj_dstr.keys())
            if len(samples) > 1:
                for u in samples[1:]:
                    lj_dstr = u.lj_distribution()
                    values = [v1+v2 for (v1,v2) in zip(values,lj_dstr.values())]
                values = [v / len(samples) for v in values]
            if len(values) == len(keys):
                ax.plot(keys, values, label="bm", color="red", linewidth=2, linestyle='-')

            if ref and ref_exists:
                lj_dstr = samples[0].lj_distribution(ref=True)
                values_ref = list(lj_dstr.values())
                keys = list(lj_dstr.keys())
                if len(self.samples) > 1:
                    for u in samples[1:]:
                        lj_dstr = u.lj_distribution(ref=True)
                        values_ref = [v1 + v2 for (v1, v2) in zip(values_ref, lj_dstr.values())]
                    values_ref = [v / len(samples) for v in values_ref]
                if len(values_ref) == len(keys):
                    ax.plot(keys, values_ref, label="ref", color="black", linewidth=2, linestyle='-')

                with open(folder+"jsd_"+name+tag+".txt", 'w') as f:
                    jsd = self.jsd(values_ref, values)
                    f.write('{} {}\n'.format("lj", jsd))
                    if np.isnan(jsd):
                        tot_score += 1.0
                    else:
                        tot_score += jsd

                results = np.array([keys, values, values_ref])
            else:
                results = np.array([keys, values])
            np.savetxt(data_folder+"lj_"+name, np.transpose(results, [1,0]))

            ax.set_xlabel('E [kJ/mol]')
            ax.set_ylabel('p')
            plt.legend()
            plt.tight_layout()
            plt.savefig(folder+"lj_"+name+tag+".pdf")



            #lj CARBONS
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(1, 1, 1)
            ax.set_title("LJ", fontsize=4)

            lj_dstr = samples[0].lj_distribution(species=['C', 'C_AR'])
            values = list(lj_dstr.values())
            keys = list(lj_dstr.keys())
            if len(samples) > 1:
                for u in samples[1:]:
                    lj_dstr = u.lj_distribution(species=['C', 'C_AR'])
                    values = [v1+v2 for (v1,v2) in zip(values,lj_dstr.values())]
                values = [v / len(samples) for v in values]
            if len(values) == len(keys):
                ax.plot(keys, values, label="bm", color="red", linewidth=2, linestyle='-')

            if ref and ref_exists:
                lj_dstr = samples[0].lj_distribution(species=['C', 'C_AR'], ref=True)
                values_ref = list(lj_dstr.values())
                keys = list(lj_dstr.keys())
                if len(self.samples) > 1:
                    for u in samples[1:]:
                        lj_dstr = u.lj_distribution(species=['C', 'C_AR'], ref=True)
                        values_ref = [v1 + v2 for (v1, v2) in zip(values_ref, lj_dstr.values())]
                    values_ref = [v / len(samples) for v in values_ref]
                if len(values_ref) == len(keys):
                    ax.plot(keys, values_ref, label="ref", color="black", linewidth=2, linestyle='-')

                with open(folder+"jsd_"+name+tag+".txt", 'a') as f:
                    f.write('{} {}\n'.format("lj_carbs", self.jsd(values_ref, values)))
                results = np.array([keys, values, values_ref])
            else:
                results = np.array([keys, values])
            np.savetxt(data_folder+"lj_carbs_"+name, np.transpose(results, [1,0]))


            ax.set_xlabel('E [kJ/mol]')
            ax.set_ylabel('p')
            plt.legend()
            plt.tight_layout()
            plt.savefig(folder+"lj_carbs_"+name+tag+".pdf")


            #lj hydros
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(1, 1, 1)
            ax.set_title("LJ", fontsize=4)

            lj_dstr = samples[0].lj_distribution(species=['H', 'H_AR'])
            values = list(lj_dstr.values())
            keys = list(lj_dstr.keys())
            if len(samples) > 1:
                for u in samples[1:]:
                    lj_dstr = u.lj_distribution(species=['H', 'H_AR'])
                    values = [v1+v2 for (v1,v2) in zip(values,lj_dstr.values())]
                values = [v / len(samples) for v in values]
            if len(values) == len(keys):
                ax.plot(keys, values, label="bm", color="red", linewidth=2, linestyle='-')

            if ref and ref_exists:
                lj_dstr = samples[0].lj_distribution(species=['H', 'H_AR'], ref=True)
                values_ref = list(lj_dstr.values())
                keys = list(lj_dstr.keys())
                if len(self.samples) > 1:
                    for u in samples[1:]:
                        lj_dstr = u.lj_distribution(species=['H', 'H_AR'], ref=True)
                        values_ref = [v1 + v2 for (v1, v2) in zip(values_ref, lj_dstr.values())]
                    values_ref = [v / len(samples) for v in values_ref]
                if len(values_ref) == len(keys):
                    ax.plot(keys, values_ref, label="ref", color="black", linewidth=2, linestyle='-')

                with open(folder+"jsd_"+name+tag+".txt", 'a') as f:
                    f.write('{} {}\n'.format("lj_hydro", self.jsd(values_ref, values)))
                results = np.array([keys, values, values_ref])
            else:
                results = np.array([keys, values])
            np.savetxt(data_folder+"lj_hydro_"+name, np.transpose(results, [1,0]))


            ax.set_xlabel('E [kJ/mol]')
            ax.set_ylabel('p')
            plt.legend()
            plt.tight_layout()
            plt.savefig(folder+"lj_hydro_"+name+tag+".pdf")


            #Bonds
            bond_score = 0.0
            fig = plt.figure(figsize=(12, 12))
            n_ax = int(np.ceil(np.sqrt(len(self.ff.bond_types))))
            c = 1
            for bond_name in self.ff.bond_types.keys():
                ax = fig.add_subplot(n_ax, n_ax, c)
                ax.set_title(bond_name, fontsize=4)

                b_dstr = samples[0].bond_distribution(bond_name)
                values = list(b_dstr.values())
                keys = list(b_dstr.keys())
                if len(samples) > 1:
                    for u in samples[1:]:
                        b_dstr = u.bond_distribution(bond_name)
                        values = [v1+v2 for (v1,v2) in zip(values,b_dstr.values())]
                    values = [v / len(samples) for v in values]
                if len(values) == len(keys):
                    ax.plot(keys, values, label="bm", color="red", linewidth=2, linestyle='-')

                if ref and ref_exists:
                    b_dstr = samples[0].bond_distribution(bond_name, ref=True)
                    values_ref = list(b_dstr.values())
                    keys = list(b_dstr.keys())
                    if len(self.samples) > 1:
                        for u in samples[1:]:
                            b_dstr = u.bond_distribution(bond_name, ref=True)
                            values_ref = [v1 + v2 for (v1, v2) in zip(values_ref, b_dstr.values())]
                        values_ref = [v / len(samples) for v in values_ref]
                    if len(values_ref) == len(keys):
                        ax.plot(keys, values_ref, label="ref", color="black", linewidth=2, linestyle='-')

                    with open(folder + "jsd_" + name + tag + ".txt", 'a') as f:
                        jsd =  self.jsd(values_ref, values)
                        f.write('{} {}\n'.format(bond_name, jsd))
                        if np.isnan(jsd):
                            tot_score += 1.0
                            bond_score += 1.0
                        else:
                            tot_score += jsd
                            bond_score += jsd
                    results = np.array([keys, values, values_ref])
                else:
                    results = np.array([keys, values])
                np.savetxt(data_folder+name+"_"+str(bond_name), np.transpose(results, [1,0]))


                title = str(bond_name)
                ax.set_title(title)
                ax.set_xlabel('d [nm]')
                ax.set_ylabel('p')
                plt.legend()
                c = c+1
            plt.tight_layout()
            plt.savefig(folder+"bond_"+name+tag+".pdf")
            #plt.show()


            #Angles
            angle_score = 0.0
            fig = plt.figure(figsize=(12, 12))
            n_ax = int(np.ceil(np.sqrt(len(self.ff.angle_types))))
            c = 1
            for angle_name in self.ff.angle_types.keys():
                ax = fig.add_subplot(n_ax, n_ax, c)
                ax.set_title(angle_name, fontsize=4)

                a_dstr = samples[0].angle_distribution(angle_name)
                values = list(a_dstr.values())
                keys = list(a_dstr.keys())
                if len(samples) > 1:
                    for u in samples[1:]:
                        a_dstr = u.angle_distribution(angle_name)
                        values = [v1+v2 for (v1,v2) in zip(values,a_dstr.values())]
                    values = [v / len(samples) for v in values]
                if len(values) == len(keys):
                    ax.plot(keys, values, label="bm", color="red", linewidth=2, linestyle='-')

                if ref and ref_exists:
                    a_dstr = samples[0].angle_distribution(angle_name, ref=True)
                    values_ref = list(a_dstr.values())
                    keys = list(a_dstr.keys())
                    if len(samples) > 1:
                        for u in samples[1:]:
                            a_dstr = u.angle_distribution(angle_name, ref=True)
                            values_ref = [v1 + v2 for (v1, v2) in zip(values_ref, a_dstr.values())]
                        values_ref = [v / len(samples) for v in values_ref]
                    if len(values_ref) == len(keys):
                        ax.plot(keys, values_ref, label="ref", color="black", linewidth=2, linestyle='-')

                    with open(folder + "jsd_" + name + tag + ".txt", 'a') as f:
                        jsd = self.jsd(values_ref, values)
                        f.write('{} {}\n'.format(angle_name, jsd))
                        if np.isnan(jsd):
                            tot_score += 1.0
                            angle_score += 1.0
                        else:
                            tot_score += jsd
                            angle_score += jsd
                    results = np.array([keys, values, values_ref])
                else:
                    results = np.array([keys, values])
                np.savetxt(data_folder+name+"_"+str(angle_name), np.transpose(results, [1,0]))


                title = str(angle_name)
                ax.set_title(title)
                ax.set_xlabel('angle [°]')
                ax.set_ylabel('p')
                plt.legend()
                c = c+1
            plt.tight_layout()
            plt.savefig(folder+"angle_"+name+tag+".pdf")


            #Dihs
            dih_score = 0.0
            fig = plt.figure(figsize=(12, 12))
            n_ax = int(np.ceil(np.sqrt(len(self.ff.dih_types))))
            c = 1
            for dih_name in self.ff.dih_types.keys():
                ax = fig.add_subplot(n_ax, n_ax, c)
                ax.set_title(dih_name, fontsize=4)

                a_dstr = samples[0].dih_distribution(dih_name)
                values = list(a_dstr.values())
                keys = list(a_dstr.keys())
                if len(samples) > 1:
                    for u in samples[1:]:
                        a_dstr = u.dih_distribution(dih_name)
                        values = [v1+v2 for (v1,v2) in zip(values,a_dstr.values())]
                    values = [v / len(samples) for v in values]
                if len(values) == len(keys):
                    ax.plot(keys, values, label="bm", color="red", linewidth=2, linestyle='-')

                if ref and ref_exists:
                    a_dstr = samples[0].dih_distribution(dih_name, ref=True)
                    values_ref = list(a_dstr.values())
                    keys = list(a_dstr.keys())
                    if len(samples) > 1:
                        for u in samples[1:]:
                            a_dstr = u.dih_distribution(dih_name, ref=True)
                            values_ref = [v1 + v2 for (v1, v2) in zip(values_ref, a_dstr.values())]
                        values_ref = [v / len(samples) for v in values_ref]
                    if len(values_ref) == len(keys):
                        ax.plot(keys, values_ref, label="ref", color="black", linewidth=2, linestyle='-')

                    with open(folder + "jsd_" + name + tag + ".txt", 'a') as f:
                        jsd = self.jsd(values_ref, values)
                        f.write('{} {}\n'.format(dih_name, jsd))
                        if np.isnan(jsd):
                            tot_score += 1.0
                            dih_score += 1.0
                        else:
                            tot_score += jsd
                            dih_score += jsd
                    results = np.array([keys, values, values_ref])
                else:
                    results = np.array([keys, values])
                np.savetxt(data_folder+name+"_"+str(dih_name), np.transpose(results, [1,0]))


                title = str(dih_name)
                ax.set_title(title)
                ax.set_xlabel('dihedral [°]')
                ax.set_ylabel('p')
                plt.legend()
                c = c+1
            plt.tight_layout()
            plt.savefig(folder+"dih_"+name+tag+".pdf")


            #RDF
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(1, 1, 1)
            ax.set_title("RDF", fontsize=4)

            a_dstr = samples[0].rdf()
            values = list(a_dstr.values())
            keys = list(a_dstr.keys())
            if len(samples) > 1:
                for u in samples[1:]:
                    a_dstr = u.rdf()
                    values = [v1+v2 for (v1,v2) in zip(values,a_dstr.values())]
                values = [v / len(samples) for v in values]
            if len(values) == len(keys):
                ax.plot(keys, values, label="bm", color="red", linewidth=2, linestyle='-')

            if ref and ref_exists:
                a_dstr = samples[0].rdf(ref=True)
                values_ref = list(a_dstr.values())
                keys = list(a_dstr.keys())
                if len(samples) > 1:
                    for u in samples[1:]:
                        a_dstr = u.rdf(ref=True)
                        values_ref = [v1 + v2 for (v1, v2) in zip(values_ref, a_dstr.values())]
                    values_ref = [v / len(samples) for v in values_ref]
                if len(values_ref) == len(keys):
                    ax.plot(keys, values_ref, label="ref", color="black", linewidth=2, linestyle='-')

                    with open(folder + "jsd_" + name + tag + ".txt", 'a') as f:
                        jsd = self.jsd(values_ref, values)
                        f.write('{} {}\n'.format("rdf", jsd))
                        if np.isnan(jsd):
                            tot_score += 1.0
                        else:
                            tot_score += jsd
                results = np.array([keys, values, values_ref])
            else:
                results = np.array([keys, values])
            np.savetxt(data_folder+name+"_"+"rdf", np.transpose(results, [1,0]))


            #title = str(dih_name)
            ax.set_xlabel('r [nm]')
            ax.set_ylabel('g(r)')
            plt.legend()
            plt.tight_layout()
            plt.savefig(folder+"rdf_"+name+tag+".pdf")

            #RDF CARBS
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(1, 1, 1)
            ax.set_title("RDF", fontsize=4)

            a_dstr = samples[0].rdf(species=['C', 'C_AR'])
            values = list(a_dstr.values())
            keys = list(a_dstr.keys())
            if len(samples) > 1:
                for u in samples[1:]:
                    a_dstr = u.rdf(species=['C', 'C_AR'])
                    values = [v1+v2 for (v1,v2) in zip(values,a_dstr.values())]
                values = [v / len(samples) for v in values]
            if len(values) == len(keys):
                ax.plot(keys, values, label="bm", color="red", linewidth=2, linestyle='-')

            if ref and ref_exists:
                a_dstr = samples[0].rdf(ref=True, species=['C', 'C_AR'])
                values_ref = list(a_dstr.values())
                keys = list(a_dstr.keys())
                if len(samples) > 1:
                    for u in samples[1:]:
                        a_dstr = u.rdf(ref=True, species=['C', 'C_AR'])
                        values_ref = [v1 + v2 for (v1, v2) in zip(values_ref, a_dstr.values())]
                    values_ref = [v / len(samples) for v in values_ref]
                if len(values_ref) == len(keys):
                    ax.plot(keys, values_ref, label="ref", color="black", linewidth=2, linestyle='-')

                    with open(folder + "jsd_" + name + tag + ".txt", 'a') as f:
                        jsd = self.jsd(values_ref, values)
                        f.write('{} {}\n'.format("rdf_carbs", jsd))
                        #if np.isnan(jsd):
                        #    tot_score += 1.0
                        #else:
                        #    tot_score += jsd

                results = np.array([keys, values, values_ref])
            else:
                results = np.array([keys, values])
            np.savetxt(data_folder+name+"_"+"rdf_carbs", np.transpose(results, [1,0]))

            #title = str(dih_name)
            ax.set_xlabel('r [nm]')
            ax.set_ylabel('g(r)')
            plt.legend()
            plt.tight_layout()
            plt.savefig(folder+"rdf_carbs_"+name+tag+".pdf")

            #RDF HYDRO
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(1, 1, 1)
            ax.set_title("RDF", fontsize=4)

            a_dstr = samples[0].rdf(species=['HC', 'H_AR'])
            values = list(a_dstr.values())
            keys = list(a_dstr.keys())
            if len(samples) > 1:
                for u in samples[1:]:
                    a_dstr = u.rdf(species=['H', 'H_AR'])
                    values = [v1+v2 for (v1,v2) in zip(values,a_dstr.values())]
                values = [v / len(samples) for v in values]
            if len(values) == len(keys):
                ax.plot(keys, values, label="bm", color="red", linewidth=2, linestyle='-')

            if ref and ref_exists:
                a_dstr = samples[0].rdf(ref=True, species=['H', 'H_AR'])
                values_ref = list(a_dstr.values())
                keys = list(a_dstr.keys())
                if len(samples) > 1:
                    for u in samples[1:]:
                        a_dstr = u.rdf(ref=True, species=['H', 'H_AR'])
                        values_ref = [v1 + v2 for (v1, v2) in zip(values_ref, a_dstr.values())]
                    values_ref = [v / len(samples) for v in values_ref]
                if len(values_ref) == len(keys):
                    ax.plot(keys, values_ref, label="ref", color="black", linewidth=2, linestyle='-')

                    with open(folder + "jsd_" + name + tag + ".txt", 'a') as f:
                        jsd = self.jsd(values_ref, values)
                        f.write('{} {}\n'.format("rdf_hydro", jsd))
                        #if np.isnan(jsd):
                        #    tot_score += 1.0
                        #else:
                        #    tot_score += jsd

                results = np.array([keys, values, values_ref])
            else:
                results = np.array([keys, values])
            np.savetxt(data_folder+name+"_"+"rdf_hydro", np.transpose(results, [1,0]))

            #title = str(dih_name)
            ax.set_xlabel('r [nm]')
            ax.set_ylabel('g(r)')
            plt.legend()
            plt.tight_layout()
            plt.savefig(folder+"rdf_hydro_"+name+tag+".pdf")


            plt.close('all')


            with open(folder + "jsd_" + name + tag + ".txt", 'a') as f:
                f.write('{} {}\n'.format("bond_score", bond_score))
                f.write('{} {}\n'.format("angle_score", angle_score))
                f.write('{} {}\n'.format("dih_score", dih_score))
                f.write('{} {}\n'.format("tot_score", tot_score))


#u = Data(["./data/sPS_t568_small"], "./forcefield/ff.txt", align=True, aug=False)

#print(u.collection[0].energy())

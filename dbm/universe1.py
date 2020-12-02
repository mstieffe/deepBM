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
from dbm.features import *
from dbm.mol import *
#from utils import *
from copy import deepcopy
#import sys
np.set_printoptions(threshold=np.inf)

class Universe():

    def __init__(self, ff, folder, cg_file, aa_file=None, cutoff=0.62, kick=0.05, align=False, aug=False, order="dfs", heavy_first=False, cg_dropout=0.0):

        start = timer()


        self.name = os.path.splitext(os.path.basename(cg_file))[0]

        self.align = align
        self.aug = aug
        self.order = order
        self.heavy_first = heavy_first
        #self.fix_seq = fix_seq

        self.cutoff_sq = cutoff*cutoff
        self.kick = kick
        self.cg_dropout = cg_dropout

        #load forcefield
        self.ff = ff

        #self.cg_file = cg_file
        #self.aa_top_file = aa_top_file
        # use MD_traj to parse gro file
        self.folder = folder
        self.cg_file = cg_file
        self.aa_file = aa_file
        cg = md.load(self.cg_file)
        if aa_file:
            aa = md.load(self.aa_file)
        # number of molecules in file
        self.n_mol = cg.topology.n_residues
        # matrix containing box dimensions
        self.box = Box(self.cg_file)

        # Go through all molecules in cg file and initialize instances of mols and beads
        #print("Setting up topology and loc environments..", end='')
        self.atoms = []
        self.beads = []
        self.mols = []
        for res in cg.topology.residues:
            self.mols.append(Mol(res.name))

            aa_top_file = os.path.join(self.folder +"/"+ res.name +"_aa.itp")
            cg_top_file = os.path.join(self.folder +"/"+ res.name +"_cg.itp")
            map_file = os.path.join(self.folder +"/"+ res.name +".map")
            env_file = os.path.join(self.folder +"/"+ res.name +".env")

            beads = []
            for bead in res.atoms:
                #type_index = [t.name for t in self.ff.bead_types].index(bead.element.symbol)
                beads.append(Bead(self.mols[-1], self.box.pbc_in_box(cg.xyz[0,bead.index]), self.ff.bead_types[bead.element.symbol]))
                self.mols[-1].add_bead(beads[-1])

            atoms = []
            for line in self.read_between("[map]", "\n", map_file):
                type_name = line.split()[1]
                #type_index = [t.name for t in self.ff.atom_types].index(type_name)
                #print(int(line.split()[2])-1)
                bead = beads[int(line.split()[2])-1]
                atoms.append(Atom(bead, self.mols[-1], bead.center, self.ff.atom_types[type_name]))
                #if np.any(self.box.pbc(aa.xyz[0, atoms[-1].index] - atoms[-1].center) != self.box.pbc3(aa.xyz[0, atoms[-1].index] - atoms[-1].center)):
                #    print(self.box.pbc(aa.xyz[0, atoms[-1].index] - atoms[-1].center))
                #    print(self.box.pbc3(aa.xyz[0, atoms[-1].index] - atoms[-1].center))

                if aa_file:
                    atoms[-1].ref_pos = self.box.pbc_diff_vec(aa.xyz[0, atoms[-1].index] - atoms[-1].center)
                bead.add_atom(atoms[-1])
                self.mols[-1].add_atom(atoms[-1])
            Atom.mol_index = 0

            #aa_edges = []
            for line in self.read_between("[bonds]", "[angles]", aa_top_file):
                index1 = int(line.split()[0])-1
                index2 = int(line.split()[1])-1
                bond = self.ff.get_bond([atoms[index1], atoms[index2]])
                if bond:
                    self.mols[-1].add_bond(bond)

            #aa_angles = []
            for line in self.read_between("[angles]", "[dihedrals]", aa_top_file):
                index1 = int(line.split()[0])-1
                index2 = int(line.split()[1])-1
                index3 = int(line.split()[2])-1
                #aa_angles.append((atoms[index1], atoms[index2], atoms[index3]))
                #aa_edges.append((atoms[index1], atoms[index2]))
                angle = self.ff.get_angle([atoms[index1], atoms[index2], atoms[index3]])
                if angle:
                    self.mols[-1].add_angle(angle)

            for line in self.read_between("[dihedrals]", "[exclusions]", aa_top_file):
                index1 = int(line.split()[0])-1
                index2 = int(line.split()[1])-1
                index3 = int(line.split()[2])-1
                index4 = int(line.split()[3])-1
                #aa_angles.append((atoms[index1], atoms[index2], atoms[index3]))
                #aa_edges.append((atoms[index1], atoms[index2]))
                dih = self.ff.get_dih([atoms[index1], atoms[index2], atoms[index3], atoms[index4]])
                if dih:
                    self.mols[-1].add_dih(dih)

            for line in self.read_between("[exclusions]", "\n", aa_top_file):
                index1 = int(line.split()[0])-1
                index2 = int(line.split()[1])-1
                self.mols[-1].add_excl([atoms[index1], atoms[index2]])

            #cg_edges = []
            for line in self.read_between("[bonds]", "[angles]", cg_top_file):
                index1 = int(line.split()[0])-1
                index2 = int(line.split()[1])-1
                #cg_edges.append((beads[index1], beads[index2]))
                self.mols[-1].add_cg_edge([beads[index1], beads[index2]])


            #add atoms and beads to universe
            self.beads += beads
            self.atoms += atoms

            #make Graph for each molecule
            self.mols[-1].make_aa_graph()
            self.mols[-1].make_cg_graph()
            #self.mols[-1].equip_cg_graph(beads, cg_edges)

            if self.align:
                for line in self.read_between("[align]", "[mult]", env_file):
                    b_index, fp_index = line.split()
                    if int(b_index) > len(self.mols[-1].beads) or int(fp_index) > len(self.mols[-1].beads):
                        raise Exception('Indices in algn section do not match the molecular structure!')
                    self.mols[-1].beads[int(b_index) - 1].fp = self.mols[-1].beads[int(fp_index) - 1]

            if self.aug:
                for line in self.read_between("[mult]", "\n", env_file):
                    b_index, m = line.split()
                    if int(b_index) > len(self.mols[-1].beads) or int(m) < 0:
                        raise Exception('Invalid number of multiples!')
                    self.mols[-1].beads[int(b_index) - 1].mult = int(m)

        Atom.index = 0
        Bead.index = 0
        Mol.index = 0
        self.n_atoms = len(self.atoms)

        print("generated mols ", timer()-start)
        # generate local envs
        self.features = {}
        self.bead_seq = []
        if self.heavy_first:
            self.atom_seq_dict_heavy = {}
            self.atom_seq_dict_hydrogens = {}
            for mol in self.mols:
                cg_seq, atom_seq_dict_heavy, atom_seq_dict_hydrogens, atom_predecessors_dict = mol.aa_seq_sep(
                    order=self.order, train=False)
                self.atom_seq_dict_heavy = {**self.atom_seq_dict_heavy, **atom_seq_dict_heavy}
                self.atom_seq_dict_hydrogens = {**self.atom_seq_dict_hydrogens, **atom_seq_dict_hydrogens}
                for bead, _ in cg_seq:
                    self.bead_seq.append(bead)
                    env_beads = self.get_loc_beads(bead)
                    for atom in atom_seq_dict_heavy[bead]:
                        predecessors = atom_predecessors_dict[atom]
                        self.features[atom] = Feature(atom, predecessors, env_beads, self.ff, self.box, self.cg_dropout)
                    for atom in atom_seq_dict_hydrogens[bead]:
                        predecessors = atom_predecessors_dict[atom]
                        self.features[atom] = Feature(atom, predecessors, env_beads, self.ff, self.box, self.cg_dropout)
        else:
            self.atom_seq_dict = {}
            for mol in self.mols:
                cg_seq, atom_seq_dict, atom_predecessors_dict = mol.aa_seq(order=self.order, train=False)
                self.atom_seq_dict = {**self.atom_seq_dict, **atom_seq_dict}
                for bead, _ in cg_seq:
                    self.bead_seq.append(bead)
                    env_beads = self.get_loc_beads(bead)
                    for atom in atom_seq_dict[bead]:
                        predecessors = atom_predecessors_dict[atom]
                        #print(mol.dihs)
                        #print(atom.type.name, atom.index)
                        #print(bead.index)
                        #print([b.index for b in env_beads])
                        self.features[atom] = Feature(atom, predecessors, env_beads, self.ff, self.box, self.cg_dropout)


        print("generated loc envs ", timer()-start)


        self.max_seq_len = 0
        self.max_beads, self.max_atoms = 0, 0
        self.max_bonds, self.max_angles, self.max_dihs, self.max_ljs = 0, 0, 0, 0
        self.max_bonds_pb, self.max_angles_pb, self.max_dihs_pb, self.max_ljs_pb = 0, 0, 0, 0
        self.update_max_values()

        #print("max", self.max_atoms, self.max_beads, self.max_bonds, self.max_angles, self.max_dihs, self.max_ljs)

        self.kick_atoms()

        print("got max values ", timer()-start)


        #print(self.energy_terms(shift=True, ref=True))
        #print("energy calc took ", timer()-start)
        #print(self.energy(shift=True, ref=True))
        #print("energy calc took ", timer()-start)

    def gen_bead_seq(self, train=False):
        bead_seq = []
        mols = self.mols[:]
        np.random.shuffle(mols)
        for mol in mols:
            bead_seq += list(zip(*mol.cg_seq(order=self.order, train=train)))[0]
        return bead_seq

    def update_max_values(self):

        for bead in self.bead_seq:
            if self.heavy_first:
                if len(self.atom_seq_dict_heavy[bead]) > self.max_seq_len: self.max_seq_len = len(self.atom_seq_dict_heavy[bead])
                if len(self.atom_seq_dict_hydrogens[bead]) > self.max_seq_len: self.max_seq_len = len(self.atom_seq_dict_hydrogens[bead])
            else:
                if len(self.atom_seq_dict[bead]) > self.max_seq_len: self.max_seq_len = len(self.atom_seq_dict[bead])
            bonds_ndx, angles_ndx, dihs_ndx, ljs_ndx = [], [], [], []
            #for atom in self.atom_seq_dict[bead]:
            for atom in bead.atoms:
                #print(atom.type.name, atom.index, bead.type.name, bead.index)
                feature = self.features[atom]
                if len(feature.env_beads) > self.max_beads: self.max_beads = len(feature.env_beads)
                if len(feature.env_atoms) > self.max_atoms: self.max_atoms = len(feature.env_atoms)
                if len(feature.bonds) > self.max_bonds: self.max_bonds = len(feature.bonds)
                if len(feature.angles) > self.max_angles: self.max_angles = len(feature.angles)
                if len(feature.dihs) > self.max_dihs: self.max_dihs = len(feature.dihs)
                if len(feature.ljs) > self.max_ljs: self.max_ljs = len(feature.ljs)
                bonds_ndx += feature.bond_ndx_init
                angles_ndx += feature.angle_ndx_init
                dihs_ndx += feature.dih_ndx_init
                ljs_ndx += feature.lj_ndx_init
            if len(bonds_ndx) > self.max_bonds_pb: self.max_bonds_pb = len(bonds_ndx)
            if len(angles_ndx) > self.max_angles_pb: self.max_angles_pb = len(angles_ndx)
            if len(dihs_ndx) > self.max_dihs_pb: self.max_dihs_pb = len(dihs_ndx)
            if len(ljs_ndx) > self.max_ljs_pb: self.max_ljs_pb = len(ljs_ndx)

    def kick_atoms(self):
        for a in self.atoms:
            a.pos = np.random.normal(-self.kick, self.kick, 3)

    def generator_combined(self, train=True, rand_rot=False, cg_kick=0.0):
        gen_init = self.generator(train=train, mode="init", rand_rot=rand_rot, cg_kick=cg_kick)
        gen_gibbs = self.generator(train=train, mode="gibbs", rand_rot=rand_rot, cg_kick=cg_kick)
        for init, gibbs in zip(gen_init, gen_gibbs):
            yield init
            yield gibbs

    def recurrent_generator_combined(self, train=True, rand_rot=False, cg_kick=0.0):
        if self.heavy_first:
            gen_init = self.recurrent_generator_heavyfirst(train=train, mode="init", rand_rot=rand_rot, cg_kick=cg_kick)
            gen_gibbs = self.recurrent_generator_heavyfirst(train=train, mode="gibbs", rand_rot=rand_rot, cg_kick=cg_kick)
        else:
            gen_init = self.recurrent_generator(train=train, mode="init", rand_rot=rand_rot, cg_kick=cg_kick)
            gen_gibbs = self.recurrent_generator(train=train, mode="gibbs", rand_rot=rand_rot, cg_kick=cg_kick)
        for init, gibbs in zip(gen_init, gen_gibbs):
            yield init
            yield gibbs

    def recurrent_generator_combined2(self, train=True, rand_rot=False, cg_kick=0.0):
        if self.heavy_first:
            gen_init = self.recurrent_generator_heavyfirst2(train=train, mode="init", rand_rot=rand_rot, cg_kick=cg_kick)
            gen_gibbs = self.recurrent_generator_heavyfirst2(train=train, mode="gibbs", rand_rot=rand_rot, cg_kick=cg_kick)
        else:
            gen_init = self.recurrent_generator(train=train, mode="init", rand_rot=rand_rot, cg_kick=cg_kick)
            gen_gibbs = self.recurrent_generator(train=train, mode="gibbs", rand_rot=rand_rot, cg_kick=cg_kick)
        for init, gibbs in zip(gen_init, gen_gibbs):
            yield init
            yield gibbs

    def generator(self, train=False, mode="init", rand_rot=False, cg_kick=0.0):
        bead_seq = self.gen_bead_seq(train=train)
        for bead in bead_seq:
            start_atom = self.atom_seq_dict[bead][0]
            feature = self.features[start_atom]

            cg_feat = self.pad2d(feature.bead_featvec, self.max_beads)
            cg_pos = self.pad2d(feature.bead_positions(cg_kick), self.max_beads)

            for atom in self.atom_seq_dict[bead]:

                if train and mode == "gibbs":
                    aa_pos = feature.atom_positions_ref()
                elif train:
                    #aa_pos = feature.atom_positions_random_mixed(0.4, self.kick)
                    aa_pos = feature.atom_positions_ref()
                else:
                    aa_pos = feature.atom_positions()
                aa_pos = self.pad2d(aa_pos, self.max_atoms)

                feature = self.features[atom]

                t_pos = feature.rot(np.array([atom.ref_pos]))
                t_type = np.zeros(self.ff.n_atom_chns)
                t_type[atom.type.channel] = 1

                if mode == "gibbs":
                    atom_feat = self.pad2d(feature.atom_featvec_gibbs, self.max_atoms)
                    bonds_ndx = feature.bond_ndx_gibbs
                    angles_ndx = feature.angle_ndx_gibbs
                    dihs_ndx = feature.dih_ndx_gibbs
                    ljs_ndx = feature.lj_ndx_gibbs
                else:
                    atom_feat = self.pad2d(feature.atom_featvec_init, self.max_atoms)
                    bonds_ndx = feature.bond_ndx_init
                    angles_ndx = feature.angle_ndx_init
                    dihs_ndx = feature.dih_ndx_init
                    ljs_ndx = feature.lj_ndx_init

                # padding for energy terms
                for n in range(0, self.max_bonds - len(bonds_ndx)):
                    bonds_ndx.append(tuple([-1, 1, 2]))
                for n in range(0, self.max_angles - len(angles_ndx)):
                    angles_ndx.append(tuple([-1, 1, 2, 3]))
                for n in range(0, self.max_dihs - len(dihs_ndx)):
                    dihs_ndx.append(tuple([-1, 1, 2, 3, 4]))
                for n in range(0, self.max_ljs - len(ljs_ndx)):
                    ljs_ndx.append(tuple([-1, 1, 2]))

                r = self.pad1d(feature.repl, self.max_atoms, value=True)
                #r = r[:, np.newaxis]

                if rand_rot and train:
                    rot_mat = self.rand_rot_mat()
                    t_pos = np.dot(t_pos, rot_mat)
                    aa_pos = np.dot(aa_pos, rot_mat)
                    cg_pos = np.dot(cg_pos, rot_mat)

                #print("jetzt kommt:")
                #for k in [t_pos, t_type, aa_pos, atom_feat, cg_pos, cg_feat, r, bonds_ndx, angles_ndx, dihs_ndx, ljs_ndx]:
                #    print(np.array(k).shape)
                if train:
                    yield t_pos, t_type, aa_pos, atom_feat, cg_pos, cg_feat, r, bonds_ndx, angles_ndx, dihs_ndx, ljs_ndx
                else:
                    yield feature, t_type, aa_pos, atom_feat, cg_pos, cg_feat
                    #yield atom, t_type, aa_pos, atom_feat, cg_pos, cg_feat

    def generator_heavyfirst(self, train=False, mode="init", rand_rot=False, cg_kick=0.0):
        bead_seq = self.gen_bead_seq(train=train)
        n_tot = len(bead_seq)
        if not train:
            bead_seq = bead_seq + bead_seq

        for bead, n in zip(bead_seq, range(0, len(bead_seq))):
            if train:
                atom_seq_dicts = [self.atom_seq_dict_heavy, self.atom_seq_dict_hydrogens]
            else:
                if n < n_tot:
                    atom_seq_dicts = [self.atom_seq_dict_heavy]
                else:
                    atom_seq_dicts = [self.atom_seq_dict_hydrogens]
            for atom_seq_dict in atom_seq_dicts:
                start_atom = atom_seq_dict[bead][0]
                feature = self.features[start_atom]

                cg_feat = self.pad2d(feature.bead_featvec, self.max_beads)
                cg_pos = self.pad2d(feature.bead_positions(cg_kick), self.max_beads)

                for atom in atom_seq_dict[bead]:

                    if train and mode == "gibbs":
                        aa_pos = feature.atom_positions_ref()
                    elif train:
                        #aa_pos = feature.atom_positions_random_mixed(0.4, self.kick)
                        aa_pos = feature.atom_positions_ref()
                    else:
                        aa_pos = feature.atom_positions()
                    aa_pos = self.pad2d(aa_pos, self.max_atoms)

                    feature = self.features[atom]

                    t_pos = feature.rot(np.array([atom.ref_pos]))
                    t_type = np.zeros(self.ff.n_atom_chns)
                    t_type[atom.type.channel] = 1


                    if mode == "gibbs":
                        atom_feat = self.pad2d(feature.atom_featvec_gibbs_hf, self.max_atoms)
                        bonds_ndx = feature.bond_ndx_gibbs
                        angles_ndx = feature.angle_ndx_gibbs
                        dihs_ndx = feature.dih_ndx_gibbs
                        ljs_ndx = feature.lj_ndx_gibbs
                    else:
                        atom_feat = self.pad2d(feature.atom_featvec_init, self.max_atoms)
                        bonds_ndx = feature.bond_ndx_init
                        angles_ndx = feature.angle_ndx_init
                        dihs_ndx = feature.dih_ndx_init
                        ljs_ndx = feature.lj_ndx_init

                    # padding for energy terms
                    for n in range(0, self.max_bonds - len(bonds_ndx)):
                        bonds_ndx.append(tuple([-1, 1, 2]))
                    for n in range(0, self.max_angles - len(angles_ndx)):
                        angles_ndx.append(tuple([-1, 1, 2, 3]))
                    for n in range(0, self.max_dihs - len(dihs_ndx)):
                        dihs_ndx.append(tuple([-1, 1, 2, 3, 4]))
                    for n in range(0, self.max_ljs - len(ljs_ndx)):
                        ljs_ndx.append(tuple([-1, 1, 2]))

                    r = self.pad1d(feature.repl, self.max_atoms, value=True)
                    #r = r[:, np.newaxis]

                    if rand_rot and train:
                        rot_mat = self.rand_rot_mat()
                        t_pos = np.dot(t_pos, rot_mat)
                        aa_pos = np.dot(aa_pos, rot_mat)
                        cg_pos = np.dot(cg_pos, rot_mat)

                    #print("jetzt kommt:")
                    #for k in [t_pos, t_type, aa_pos, atom_feat, cg_pos, cg_feat, r, bonds_ndx, angles_ndx, dihs_ndx, ljs_ndx]:
                    #    print(np.array(k).shape)
                    if train:
                        yield t_pos, t_type, aa_pos, atom_feat, cg_pos, cg_feat, r, bonds_ndx, angles_ndx, dihs_ndx, ljs_ndx
                    else:
                        yield feature, t_type, aa_pos, atom_feat, cg_pos, cg_feat
                        #yield atom, t_type, aa_pos, atom_feat, cg_pos, cg_feat

    def recurrent_generator(self, train=False, mode="init", rand_rot=False, cg_kick=0.0):
        bead_seq = self.gen_bead_seq(train=train)
        for bead in bead_seq:
            start_atom = self.atom_seq_dict[bead][0]
            feature = self.features[start_atom]

            cg_feat = self.pad2d(feature.bead_featvec, self.max_beads)
            cg_pos = self.pad2d(feature.bead_positions(cg_kick), self.max_beads)

            if train and mode == "gibbs":
                aa_pos = feature.atom_positions_ref()
            elif train:
                #aa_pos = feature.atom_positions_random_mixed(0.4, self.kick)
                aa_pos = feature.atom_positions_ref()
            else:
                aa_pos = feature.atom_positions()
            aa_pos = self.pad2d(aa_pos, self.max_atoms)

            target_pos, target_type, aa_feat, repl = [], [], [], []
            bonds_ndx, angles_ndx, dihs_ndx, ljs_ndx = [], [], [], []
            for atom in self.atom_seq_dict[bead]:
                feature = self.features[atom]

                if train:
                    t_pos = feature.rot(np.array([atom.ref_pos]))
                    target_pos.append(t_pos)
                else:
                    target_pos.append(np.zeros((1, 3)))
                t_type = np.zeros(self.ff.n_atom_chns)
                t_type[atom.type.channel] = 1
                target_type.append(t_type)

                if mode == "gibbs":
                    atom_featvec = self.pad2d(feature.atom_featvec_gibbs, self.max_atoms)
                    bonds_ndx += feature.bond_ndx_gibbs
                    angles_ndx += feature.angle_ndx_gibbs
                    dihs_ndx += feature.dih_ndx_gibbs
                    ljs_ndx += feature.lj_ndx_gibbs
                else:
                    atom_featvec = self.pad2d(feature.atom_featvec_init, self.max_atoms)
                    bonds_ndx += feature.bond_ndx_init
                    angles_ndx += feature.angle_ndx_init
                    dihs_ndx += feature.dih_ndx_init
                    ljs_ndx += feature.lj_ndx_init
                aa_feat.append(atom_featvec)

                r = self.pad1d(feature.repl, self.max_atoms, value=True)
                #r = r[np.newaxis, np.newaxis, np.newaxis, :]
                repl.append(r)


            mask = np.zeros(self.max_seq_len)
            mask[:len(self.atom_seq_dict[bead])] = 1

            # Padding for recurrent training
            for n in range(0, self.max_seq_len - len(self.atom_seq_dict[bead])):
                target_pos.append(np.zeros((1, 3)))
                target_type.append(target_type[-1])
                aa_feat.append(np.zeros(aa_feat[-1].shape))
                repl.append(np.ones(repl[-1].shape, dtype=bool))

            # remove duplicates in energy ndx
            bonds_ndx = list(set(bonds_ndx))
            angles_ndx = list(set(angles_ndx))
            dihs_ndx = list(set(dihs_ndx))
            ljs_ndx = list(set(ljs_ndx))

            # padding for energy terms
            for n in range(0, self.max_bonds_pb - len(bonds_ndx)):
                bonds_ndx.append(tuple([-1, 1, 2]))
            for n in range(0, self.max_angles_pb - len(angles_ndx)):
                angles_ndx.append(tuple([-1, 1, 2, 3]))
            for n in range(0, self.max_dihs_pb - len(dihs_ndx)):
                dihs_ndx.append(tuple([-1, 1, 2, 3, 4]))
            for n in range(0, self.max_ljs_pb - len(ljs_ndx)):
                ljs_ndx.append(tuple([-1, 1, 2]))

            if rand_rot and train:
                rot_mat = self.rand_rot_mat()
                target_pos = np.dot(target_pos, rot_mat)
                aa_pos = np.dot(aa_pos, rot_mat)
                cg_pos = np.dot(cg_pos, rot_mat)

            yield self.atom_seq_dict[bead], target_pos, target_type, aa_feat, repl, mask, aa_pos, cg_pos, cg_feat, bonds_ndx, angles_ndx, dihs_ndx, ljs_ndx

            #if train:
            #    yield target_pos, target_type, aa_feat, repl, mask, aa_pos, cg_pos, cg_feat, bonds_ndx, angles_ndx, dihs_ndx, ljs_ndx
            #else:
            #    yield self.atom_seq_dict[bead], target_type, aa_feat, repl, aa_pos, cg_pos, cg_feat, bonds_ndx, angles_ndx, dihs_ndx, ljs_ndx


    def recurrent_generator_heavyfirst(self, train=False, mode="init", rand_rot=False, cg_kick=0.0):
        bead_seq = self.gen_bead_seq(train=train)
        n_tot = len(bead_seq)
        if not train:
            bead_seq = bead_seq + bead_seq

        for bead, n in zip(bead_seq, range(0,len(bead_seq))):
            if train:
                atom_seq_dicts = [self.atom_seq_dict_heavy, self.atom_seq_dict_hydrogens]
            else:
                if n < n_tot:
                    atom_seq_dicts = [self.atom_seq_dict_heavy]
                else:
                    atom_seq_dicts = [self.atom_seq_dict_hydrogens]
            for atom_seq_dict in atom_seq_dicts:
                start_atom = atom_seq_dict[bead][0]
                feature = self.features[start_atom]

                cg_feat = self.pad2d(feature.bead_featvec, self.max_beads)
                cg_pos = self.pad2d(feature.bead_positions(cg_kick), self.max_beads)

                if train and mode == "gibbs":
                    aa_pos = feature.atom_positions_ref()
                elif train:
                    #aa_pos = feature.atom_positions_random_mixed(0.4, self.kick)
                    aa_pos = feature.atom_positions_ref()
                else:
                    aa_pos = feature.atom_positions()
                aa_pos = self.pad2d(aa_pos, self.max_atoms)

                target_pos, target_type, aa_feat, repl = [], [], [], []
                bonds_ndx, angles_ndx, dihs_ndx, ljs_ndx = [], [], [], []
                for atom in atom_seq_dict[bead]:
                    feature = self.features[atom]

                    if train:
                        t_pos = feature.rot(np.array([atom.ref_pos]))
                        target_pos.append(t_pos)
                    else:
                        target_pos.append(np.zeros((1, 3)))
                    t_type = np.zeros(self.ff.n_atom_chns)
                    t_type[atom.type.channel] = 1
                    target_type.append(t_type)

                    if mode == "gibbs":
                        atom_featvec = self.pad2d(feature.atom_featvec_gibbs_hf, self.max_atoms)
                        bonds_ndx += feature.bond_ndx_gibbs
                        angles_ndx += feature.angle_ndx_gibbs
                        dihs_ndx += feature.dih_ndx_gibbs
                        ljs_ndx += feature.lj_ndx_gibbs
                    else:
                        atom_featvec = self.pad2d(feature.atom_featvec_init, self.max_atoms)
                        bonds_ndx += feature.bond_ndx_init
                        angles_ndx += feature.angle_ndx_init
                        dihs_ndx += feature.dih_ndx_init
                        ljs_ndx += feature.lj_ndx_init

                    aa_feat.append(atom_featvec)

                    r = self.pad1d(feature.repl, self.max_atoms, value=True)
                    #r = r[np.newaxis, np.newaxis, np.newaxis, :]
                    repl.append(r)

                mask = np.zeros(self.max_seq_len)
                mask[:len(atom_seq_dict[bead])] = 1

                # Padding for recurrent training
                for n in range(0, self.max_seq_len - len(atom_seq_dict[bead])):
                    target_pos.append(np.zeros((1, 3)))
                    target_type.append(target_type[-1])
                    aa_feat.append(np.zeros(aa_feat[-1].shape))
                    repl.append(np.ones(repl[-1].shape, dtype=bool))

                # remove duplicates in energy ndx
                bonds_ndx = list(set(bonds_ndx))
                angles_ndx = list(set(angles_ndx))
                dihs_ndx = list(set(dihs_ndx))
                ljs_ndx = list(set(ljs_ndx))

                # padding for energy terms
                for n in range(0, self.max_bonds_pb - len(bonds_ndx)):
                    bonds_ndx.append(tuple([-1, 1, 2]))
                for n in range(0, self.max_angles_pb - len(angles_ndx)):
                    angles_ndx.append(tuple([-1, 1, 2, 3]))
                for n in range(0, self.max_dihs_pb - len(dihs_ndx)):
                    dihs_ndx.append(tuple([-1, 1, 2, 3, 4]))
                for n in range(0, self.max_ljs_pb - len(ljs_ndx)):
                    ljs_ndx.append(tuple([-1, 1, 2]))

                if rand_rot and train:
                    rot_mat = self.rand_rot_mat()
                    target_pos = np.dot(target_pos, rot_mat)
                    aa_pos = np.dot(aa_pos, rot_mat)
                    cg_pos = np.dot(cg_pos, rot_mat)

                yield atom_seq_dict[bead], target_pos, target_type, aa_feat, repl, mask, aa_pos, cg_pos, cg_feat, bonds_ndx, angles_ndx, dihs_ndx, ljs_ndx

    def pad_lj_ndx(self, lj_ndx):
        for n in range(0, self.max_ljs_pb - len(lj_ndx)):
            lj_ndx.append(tuple([-1, 1, 2]))
        return lj_ndx

    def pad_bond_ndx(self, bond_ndx):
        for n in range(0, self.max_bonds_pb - len(bond_ndx)):
            bond_ndx.append(tuple([-1, 1, 2]))
        return bond_ndx

    def pad_angle_ndx(self, angle_ndx):
        for n in range(0, self.max_angles_pb - len(angle_ndx)):
            angle_ndx.append(tuple([-1, 1, 2, 3]))
        return angle_ndx

    def pad_dih_ndx(self, dih_ndx):
        for n in range(0, self.max_dihs_pb - len(dih_ndx)):
            dih_ndx.append(tuple([-1, 1, 2, 3, 4]))
        return dih_ndx

    def recurrent_generator_heavyfirst2(self, train=False, mode="init", rand_rot=False, cg_kick=0.0):
        bead_seq = self.gen_bead_seq(train=train)
        n_tot = len(bead_seq)
        if not train:
            bead_seq = bead_seq + bead_seq

        for bead, n in zip(bead_seq, range(0, len(bead_seq))):
            if train:
                atom_seq_dicts = [self.atom_seq_dict_heavy, self.atom_seq_dict_hydrogens]
            else:
                if n < n_tot:
                    atom_seq_dicts = [self.atom_seq_dict_heavy]
                else:
                    atom_seq_dicts = [self.atom_seq_dict_hydrogens]
            for atom_seq_dict in atom_seq_dicts:
                start_atom = atom_seq_dict[bead][0]
                feature = self.features[start_atom]

                cg_feat = self.pad2d(feature.bead_featvec, self.max_beads)
                cg_pos = self.pad2d(feature.bead_positions(cg_kick), self.max_beads)

                if train and mode == "gibbs":
                    aa_pos = feature.atom_positions_ref()
                elif train:
                    # aa_pos = feature.atom_positions_random_mixed(0.4, self.kick)
                    aa_pos = feature.atom_positions_ref()
                else:
                    aa_pos = feature.atom_positions()
                aa_pos = self.pad2d(aa_pos, self.max_atoms)

                target_pos, target_type, aa_feat, repl = [], [], [], []
                bonds_ndx, angles_ndx, dihs_ndx, ljs_ndx = [], [], [], []
                atomwise_bond_ndx, atomwise_angle_ndx, atomwise_dih_ndx, atomwise_lj_ndx = [], [], [], []
                for atom in atom_seq_dict[bead]:
                    feature = self.features[atom]

                    if train:
                        t_pos = feature.rot(np.array([atom.ref_pos]))
                        target_pos.append(t_pos)
                    else:
                        target_pos.append(np.zeros((1, 3)))
                    t_type = np.zeros(self.ff.n_atom_chns)
                    t_type[atom.type.channel] = 1
                    target_type.append(t_type)

                    if mode == "gibbs":
                        atom_featvec = self.pad2d(feature.atom_featvec_gibbs_hf, self.max_atoms)
                        bonds_ndx += feature.bond_ndx_gibbs
                        angles_ndx += feature.angle_ndx_gibbs
                        dihs_ndx += feature.dih_ndx_gibbs
                        ljs_ndx += feature.lj_ndx_gibbs

                        atomwise_bond_ndx.append(self.pad_bond_ndx(feature.bond_ndx_gibbs[:]))
                        atomwise_angle_ndx.append(self.pad_angle_ndx(feature.angle_ndx_gibbs[:]))
                        atomwise_dih_ndx.append(self.pad_dih_ndx(feature.dih_ndx_gibbs[:]))
                        atomwise_lj_ndx.append(self.pad_lj_ndx(feature.lj_ndx_gibbs[:]))
                    else:
                        atom_featvec = self.pad2d(feature.atom_featvec_init, self.max_atoms)
                        bonds_ndx += feature.bond_ndx_init
                        angles_ndx += feature.angle_ndx_init
                        dihs_ndx += feature.dih_ndx_init
                        ljs_ndx += feature.lj_ndx_init

                        atomwise_bond_ndx.append(self.pad_bond_ndx(feature.bond_ndx_init[:]))
                        atomwise_angle_ndx.append(self.pad_angle_ndx(feature.angle_ndx_init[:]))
                        atomwise_dih_ndx.append(self.pad_dih_ndx(feature.dih_ndx_init[:]))
                        atomwise_lj_ndx.append(self.pad_lj_ndx(feature.lj_ndx_init[:]))

                    aa_feat.append(atom_featvec)

                    r = self.pad1d(feature.repl, self.max_atoms, value=True)
                    # r = r[np.newaxis, np.newaxis, np.newaxis, :]
                    repl.append(r)

                mask = np.zeros(self.max_seq_len)
                mask[:len(atom_seq_dict[bead])] = 1

                # Padding for recurrent training
                for n in range(0, self.max_seq_len - len(atom_seq_dict[bead])):
                    target_pos.append(np.zeros((1, 3)))
                    target_type.append(target_type[-1])
                    aa_feat.append(np.zeros(aa_feat[-1].shape))
                    repl.append(np.ones(repl[-1].shape, dtype=bool))
                    atomwise_bond_ndx.append([tuple([-1, 1, 2])]*self.max_bonds_pb)
                    atomwise_angle_ndx.append([tuple([-1, 1, 2, 3])]*self.max_angles_pb)
                    atomwise_dih_ndx.append([tuple([-1, 1, 2, 3, 4])]*self.max_dihs_pb)
                    atomwise_lj_ndx.append([tuple([-1, 1, 2])]*self.max_ljs_pb)

                # remove duplicates in energy ndx
                bonds_ndx = list(set(bonds_ndx))
                angles_ndx = list(set(angles_ndx))
                dihs_ndx = list(set(dihs_ndx))
                ljs_ndx = list(set(ljs_ndx))

                # padding for energy terms
                for n in range(0, self.max_bonds_pb - len(bonds_ndx)):
                    bonds_ndx.append(tuple([-1, 1, 2]))
                for n in range(0, self.max_angles_pb - len(angles_ndx)):
                    angles_ndx.append(tuple([-1, 1, 2, 3]))
                for n in range(0, self.max_dihs_pb - len(dihs_ndx)):
                    dihs_ndx.append(tuple([-1, 1, 2, 3, 4]))
                for n in range(0, self.max_ljs_pb - len(ljs_ndx)):
                    ljs_ndx.append(tuple([-1, 1, 2]))

                if rand_rot and train:
                    rot_mat = self.rand_rot_mat()
                    target_pos = np.dot(target_pos, rot_mat)
                    aa_pos = np.dot(aa_pos, rot_mat)
                    cg_pos = np.dot(cg_pos, rot_mat)

                yield atom_seq_dict[bead], target_pos, target_type, aa_feat, repl, mask, aa_pos, cg_pos, cg_feat, atomwise_bond_ndx, atomwise_angle_ndx, atomwise_dih_ndx, atomwise_lj_ndx, bonds_ndx, angles_ndx, dihs_ndx, ljs_ndx



                #if train:
                #    yield target_pos, target_type, aa_feat, repl, mask, aa_pos, cg_pos, cg_feat, bonds_ndx, angles_ndx, dihs_ndx, ljs_ndx
                #else:
                #    yield atom_seq_dict[bead], target_type, aa_feat, repl, aa_pos, cg_pos, cg_feat, bonds_ndx, angles_ndx, dihs_ndx, ljs_ndx


    def pad2d(self, vec, max, value=0):
        vec = np.pad(vec, ((0, max - len(vec)), (0, 0)), 'constant', constant_values=(0, value))
        return vec

    def pad1d(self, vec, max, value=0):
        vec = np.pad(vec, (0, max - len(vec)), 'constant', constant_values=(0, value))
        return vec

    def get_loc_beads(self, bead):
        centered_positions = np.array([b.center for b in self.beads]) - bead.center
        centered_positions = np.array([self.box.pbc_diff_vec(pos) for pos in centered_positions])
        #centered_positions = self.box.pbc_diff_vec_batch(np.array(centered_positions))
        centered_positions_sq = [r[0] * r[0] + r[1] * r[1] + r[2] * r[2] for r in centered_positions]

        indices = np.where(np.array(centered_positions_sq) <= self.cutoff_sq)[0]
        return [self.beads[i] for i in indices]

    def rand_rot_mat(self):
        #rotation axis
        if self.align:
            v_rot = np.array([0.0, 0.0, 1.0])
        else:
            phi = np.random.uniform(0, np.pi * 2)
            costheta = np.random.uniform(-1, 1)
            theta = np.arccos(costheta)
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            v_rot = np.array([x, y, z])

        #rotation angle
        theta = np.random.uniform(0, np.pi * 2)

        #rotation matrix
        a = math.cos(theta / 2.0)
        b, c, d = -v_rot * math.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        rot_mat = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                         [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                         [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

        return rot_mat

    def read_between(self, start, end, file):
        #generator to yield line between start and end
        file = open(file)
        rec = False
        for line in file:
            if line.startswith(";") or line.startswith("\n"):
                continue
            if not rec:
                if line.startswith(start):
                    rec = True
            elif line.startswith(end):
                rec = False
            else:
                yield line
        file.close()



    def energy(self, ref=False, shift=False, resolve_terms=False):
        pos1, pos2, equil, f_c = [], [], [], []
        for mol in self.mols:
            if ref:
                pos1 += [bond.atoms[0].ref_pos + bond.atoms[0].center for bond in mol.bonds]
                pos2 += [bond.atoms[1].ref_pos + bond.atoms[1].center for bond in mol.bonds]
            else:
                pos1 += [bond.atoms[0].pos + bond.atoms[0].center for bond in mol.bonds]
                pos2 += [bond.atoms[1].pos + bond.atoms[1].center for bond in mol.bonds]
            equil += [bond.type.equil for bond in mol.bonds]
            f_c += [bond.type.force_const for bond in mol.bonds]
        dis = self.box.pbc_diff_vec_batch(np.array(pos1) - np.array(pos2))
        dis = np.sqrt(np.sum(np.square(dis), axis=-1))
        bond_energy = self.ff.bond_energy(dis, np.array(equil), np.array(f_c))

        pos1, pos2, pos3, equil, f_c = [], [], [], [], []
        for mol in self.mols:
            if ref:
                pos1 += [angle.atoms[0].ref_pos + angle.atoms[0].center for angle in mol.angles]
                pos2 += [angle.atoms[1].ref_pos + angle.atoms[1].center for angle in mol.angles]
                pos3 += [angle.atoms[2].ref_pos + angle.atoms[2].center for angle in mol.angles]
            else:
                pos1 += [angle.atoms[0].pos + angle.atoms[0].center for angle in mol.angles]
                pos2 += [angle.atoms[1].pos + angle.atoms[1].center for angle in mol.angles]
                pos3 += [angle.atoms[2].pos + angle.atoms[2].center for angle in mol.angles]
            equil += [angle.type.equil for angle in mol.angles]
            f_c += [angle.type.force_const for angle in mol.angles]
        vec1 = self.box.pbc_diff_vec_batch(np.array(pos1) - np.array(pos2))
        vec2 = self.box.pbc_diff_vec_batch(np.array(pos3) - np.array(pos2))
        angle_energy = self.ff.angle_energy(vec1, vec2, equil, f_c)

        pos1, pos2, pos3, pos4, func, mult, equil, f_c = [], [], [], [], [], [], [], []
        for mol in self.mols:
            if ref:
                pos1 += [dih.atoms[0].ref_pos + dih.atoms[0].center for dih in mol.dihs]
                pos2 += [dih.atoms[1].ref_pos + dih.atoms[1].center for dih in mol.dihs]
                pos3 += [dih.atoms[2].ref_pos + dih.atoms[2].center for dih in mol.dihs]
                pos4 += [dih.atoms[3].ref_pos + dih.atoms[3].center for dih in mol.dihs]
            else:
                pos1 += [dih.atoms[0].pos + dih.atoms[0].center for dih in mol.dihs]
                pos2 += [dih.atoms[1].pos + dih.atoms[1].center for dih in mol.dihs]
                pos3 += [dih.atoms[2].pos + dih.atoms[2].center for dih in mol.dihs]
                pos4 += [dih.atoms[3].pos + dih.atoms[3].center for dih in mol.dihs]
            func += [angle.type.func for angle in mol.dihs]
            mult += [angle.type.mult for angle in mol.dihs]
            equil += [angle.type.equil for angle in mol.dihs]
            f_c += [angle.type.force_const for angle in mol.dihs]
        vec1 = self.box.pbc_diff_vec_batch(np.array(pos2) - np.array(pos1))
        vec2 = self.box.pbc_diff_vec_batch(np.array(pos2) - np.array(pos3))
        vec3 = self.box.pbc_diff_vec_batch(np.array(pos4) - np.array(pos3))
        dih_energy = self.ff.dih_energy(vec1, vec2, vec3, func, mult, equil, f_c)

        lj_energy = 0.0
        for a in self.atoms:
            lj_energy += self.features[a].lj_energy(ref=ref, shift=shift)
        lj_energy = lj_energy / 2

        if resolve_terms:
            return {
                "bond": bond_energy,
                "angle": angle_energy,
                "dih": dih_energy,
                "lj": lj_energy
            }
        else:
            return bond_energy + angle_energy + dih_energy + lj_energy


    def lj_distribution_bead(self, bead_ndx, n_bins=40, low=-800, high=400.0, ref=False):
        lj_dstr = {}
        lj = []
        for mol in self.mols:
            e = 0.0
            for atom in mol.beads[bead_ndx].atoms:
                if ref:
                    e_intra, e_inter = self.features[atom].lj_energy_intra_inter(ref=True)
                else:
                    e_intra, e_inter = self.features[atom].lj_energy_intra_inter(ref=False)
                e += e_intra/2.0 + e_inter
            lj.append(e)
        #print(len(pos1))
        if lj != []:
            hist, _ = np.histogram(lj, bins=n_bins, range=(low, high), normed=False, density=False)
        else:
            hist = np.zeros(n_bins)

        dr = float((high-low) / n_bins)
        for i in range(0, n_bins):
            lj_dstr[low + (i + 0.5) * dr] = hist[i]
        return lj_dstr

    def lj_distribution(self, species=None, n_bins=80, low=-600, high=400.0, ref=False):
        lj_dstr = {}
        lj = []
        for mol in self.mols:
            e = 0.0
            if species:
                atoms = [a for a in mol.atoms if a.type.name in species]
            else:
                atoms = mol.atoms
            for atom in atoms:
                if ref:
                    e_intra, e_inter = self.features[atom].lj_energy_intra_inter(ref=True)
                else:
                    e_intra, e_inter = self.features[atom].lj_energy_intra_inter(ref=False)
                e += e_intra/2.0 + e_inter
            lj.append(e)
        #print(len(pos1))
        if lj != []:
            hist, _ = np.histogram(lj, bins=n_bins, range=(low, high), normed=False, density=False)
        else:
            hist = np.zeros(n_bins)

        dr = float((high-low) / n_bins)
        for i in range(0, n_bins):
            lj_dstr[low + (i + 0.5) * dr] = hist[i]
        return lj_dstr


    def plot_bond_dstr(self, bond_name, n_bins=80, low=0.0, high=0.2, ref=False):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)

        if ref:
            dstr_ref = self.bond_distribution(bond_name, n_bins=n_bins, low=low, high=high, ref=True)
            ax.plot(list(dstr_ref), list(dstr_ref.values()), label="reference", color="black", linewidth=2, linestyle='-')

        dstr = self.bond_distribution(bond_name, n_bins=n_bins, low=low, high=high, ref=False)
        ax.plot(list(dstr), list(dstr.values()), label="bm", color="red", linewidth=2, linestyle='-')

        ax.set_xlim([low, high])
        #ax.set_ylim([0.0, 1.50])

        title = 'bond distribution ' + str(bond_name)
        ax.set_title(title)
        ax.set_xlabel('d [nm]')
        ax.set_ylabel('p')
        plt.legend()
        plt.show()


    def bond_distribution(self, bond_name, n_bins=80, low=0.0, high=0.2, ref=False):
        bond_dstr = {}
        pos1, pos2 = [], []
        for mol in self.mols:
            if ref:
                pos1 += [bond.atoms[0].ref_pos + bond.atoms[0].center for bond in mol.bonds if bond.type.name == bond_name]
                pos2 += [bond.atoms[1].ref_pos + bond.atoms[1].center for bond in mol.bonds if bond.type.name == bond_name]
            else:
                pos1 += [bond.atoms[0].pos + bond.atoms[0].center for bond in mol.bonds if bond.type.name == bond_name]
                pos2 += [bond.atoms[1].pos + bond.atoms[1].center for bond in mol.bonds if bond.type.name == bond_name]
        #print(len(pos1))
        if pos1 != []:
            dis = self.box.pbc_diff_vec_batch(np.array(pos1) - np.array(pos2))
            dis = np.sqrt(np.sum(np.square(dis), axis=-1))
            hist, _ = np.histogram(dis, bins=n_bins, range=(low, high), normed=False, density=False)
        else:
            hist = np.zeros(n_bins)

        dr = float((high-low) / n_bins)
        for i in range(0, n_bins):
            bond_dstr[low + (i + 0.5) * dr] = hist[i]
        return bond_dstr

    def plot_angle_dstr(self, angletype, n_bins=80, low=70.0, high=150., ref=False):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)

        if ref:
            dstr_ref = self.angle_distribution(angletype, n_bins=n_bins, low=low, high=high, ref=True)
            ax.plot(list(dstr_ref), list(dstr_ref.values()), label="reference", color="black", linewidth=2, linestyle='-')

        dstr = self.angle_distribution(angletype, n_bins=n_bins, low=low, high=high, ref=False)
        ax.plot(list(dstr), list(dstr.values()), label="bm", color="red", linewidth=2, linestyle='-')

        ax.set_xlim([low, high])
        #ax.set_ylim([0.0, 1.50])

        title = 'angle distribution ' + str(angletype.name)
        ax.set_title(title)
        ax.set_xlabel('angle [°]')
        ax.set_ylabel('p')
        plt.legend()
        plt.show()

    def angle_distribution(self, angle_name, n_bins=80, low=70.0, high=150., ref=False):
        angle_dstr = {}
        pos1, pos2, pos3 = [], [], []
        for mol in self.mols:
            if ref:
                pos1 += [angle.atoms[0].ref_pos + angle.atoms[0].center for angle in mol.angles if angle.type.name == angle_name]
                pos2 += [angle.atoms[1].ref_pos + angle.atoms[1].center for angle in mol.angles if angle.type.name == angle_name]
                pos3 += [angle.atoms[2].ref_pos + angle.atoms[2].center for angle in mol.angles if angle.type.name == angle_name]

            else:
                pos1 += [angle.atoms[0].pos + angle.atoms[0].center for angle in mol.angles if angle.type.name == angle_name]
                pos2 += [angle.atoms[1].pos + angle.atoms[1].center for angle in mol.angles if angle.type.name == angle_name]
                pos3 += [angle.atoms[2].pos + angle.atoms[2].center for angle in mol.angles if angle.type.name == angle_name]
        if pos1 != []:
            vec1 = self.box.pbc_diff_vec_batch(np.array(pos1) - np.array(pos2))
            vec2 = self.box.pbc_diff_vec_batch(np.array(pos3) - np.array(pos2))
            norm1 = np.square(vec1)
            norm1 = np.sum(norm1, axis=-1)
            norm1 = np.sqrt(norm1)
            norm2 = np.square(vec2)
            norm2 = np.sum(norm2, axis=-1)
            norm2 = np.sqrt(norm2)
            norm = np.multiply(norm1, norm2)
            dot = np.multiply(vec1, vec2)
            dot = np.sum(dot, axis=-1)
            a = np.clip(np.divide(dot, norm), -1.0, 1.0)
            a = np.arccos(a)
            a = a*180./math.pi
            hist, bin_edges = np.histogram(a, bins=n_bins, range=(low, high), normed=False, density=False)
        else:
            hist = np.zeros(n_bins)

        dr = float((high-low) / n_bins)
        for i in range(0, n_bins):
            angle_dstr[low + (i + 0.5) * dr] = hist[i]
        return angle_dstr

    def plot_dih_dstr(self, dihtype, n_bins=80, low=0.0, high=360., ref=False):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)

        if ref:
            dstr_ref = self.dih_distribution(dihtype, n_bins=n_bins, low=low, high=high, ref=True)
            ax.plot(list(dstr_ref), list(dstr_ref.values()), label="reference", color="black", linewidth=2, linestyle='-')

        dstr = self.dih_distribution(dihtype, n_bins=n_bins, low=low, high=high, ref=False)
        ax.plot(list(dstr), list(dstr.values()), label="bm", color="red", linewidth=2, linestyle='-')

        ax.set_xlim([low, high])
        #ax.set_ylim([0.0, 1.50])

        title = 'dihedral distribution ' + str(dihtype.name)
        ax.set_title(title)
        ax.set_xlabel('angle [°]')
        ax.set_ylabel('p')
        plt.legend()
        plt.show()

    def dih_distribution(self, dih_name, n_bins=80, low=0.0, high=360., ref=False):
        dih_dstr = {}
        pos1, pos2, pos3, pos4 = [], [], [], []
        for mol in self.mols:
            if ref:
                pos1 += [dih.atoms[0].ref_pos + dih.atoms[0].center for dih in mol.dihs if dih.type.name == dih_name]
                pos2 += [dih.atoms[1].ref_pos + dih.atoms[1].center for dih in mol.dihs if dih.type.name == dih_name]
                pos3 += [dih.atoms[2].ref_pos + dih.atoms[2].center for dih in mol.dihs if dih.type.name == dih_name]
                pos4 += [dih.atoms[3].ref_pos + dih.atoms[3].center for dih in mol.dihs if dih.type.name == dih_name]

            else:
                pos1 += [dih.atoms[0].pos + dih.atoms[0].center for dih in mol.dihs if dih.type.name == dih_name]
                pos2 += [dih.atoms[1].pos + dih.atoms[1].center for dih in mol.dihs if dih.type.name == dih_name]
                pos3 += [dih.atoms[2].pos + dih.atoms[2].center for dih in mol.dihs if dih.type.name == dih_name]
                pos4 += [dih.atoms[3].pos + dih.atoms[3].center for dih in mol.dihs if dih.type.name == dih_name]

        if pos1 != []:
            vec1 = self.box.pbc_diff_vec_batch(np.array(pos2) - np.array(pos1))
            vec2 = self.box.pbc_diff_vec_batch(np.array(pos2) - np.array(pos3))
            vec3 = self.box.pbc_diff_vec_batch(np.array(pos4) - np.array(pos3))
            plane1 = np.cross(vec1, vec2)
            plane2 = np.cross(vec2, vec3)
            norm1 = np.square(plane1)
            norm1 = np.sum(norm1, axis=-1)
            norm1 = np.sqrt(norm1)
            norm2 = np.square(plane2)
            norm2 = np.sum(norm2, axis=-1)
            norm2 = np.sqrt(norm2)
            norm = np.multiply(norm1, norm2)
            dot = np.multiply(plane1, plane2)
            dot = np.sum(dot, axis=-1)
            a = np.clip(np.divide(dot, norm), -1.0, 1.0)
            a = np.arccos(a)
            a = a*180./math.pi
            hist, bin_edges = np.histogram(a, bins=n_bins, range=(low, high), normed=False, density=False)
        else:
            hist = np.zeros(n_bins)

        dr = float((high-low) / n_bins)
        for i in range(0, n_bins):
            dih_dstr[low + (i + 0.5) * dr] = hist[i]
        return dih_dstr

    def plot_rdf(self, n_bins=40, max_dist=1.2, species=None, ref=False, excl=2):

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)

        if ref:
            rdf_ref = self.rdf(n_bins, max_dist, species=species, ref=True, excl=excl)
            ax.plot(list(rdf_ref), list(rdf_ref.values()), label="reference", color="black", linewidth=2, linestyle='-')

        rdf = self.rdf(n_bins, max_dist, species=species, ref=False, excl=excl)
        ax.plot(list(rdf), list(rdf.values()), label="bm", color="red", linewidth=2, linestyle='-')

        ax.set_xlim([0.0, max_dist])
        ax.set_ylim([0.0, 1.50])

        ax.set_title('Radial Distribution Function ')
        ax.set_xlabel('r [nm]')
        ax.set_ylabel('g(r)')
        plt.axhline(y=1.0, color='grey', linestyle='--')
        ax.text(0.8, 0.1, "atomtypes: "+str(species), fontsize=10)
        plt.legend()
        #plt.savefig(out_file, bbox_inches='tight')
        plt.show()

    def rdf(self, n_bins=40, max_dist=1.2, species=None, ref=False, excl=3):
        rdf = {}
        dr = float(max_dist / n_bins)
        hist = [0]*n_bins
        if species:
            atoms = [a for a in self.atoms if a.type.name in species]
        else:
            atoms = self.atoms
        n_atoms = len(atoms)

        if ref:
            x = np.array([self.box.pbc_in_box(a.ref_pos + a.center) for a in atoms])
        else:
            x = np.array([self.box.pbc_in_box(a.pos + a.center) for a in atoms])

        if atoms != []:
            d = x[:, np.newaxis, :] - x[np.newaxis, :, :]
            d = np.reshape(d, (n_atoms*n_atoms, 3))
            d = self.box.pbc_diff_vec_batch(d)
            d = np.reshape(d, (n_atoms, n_atoms, 3))
            d = np.sqrt(np.sum(d ** 2, axis=-1))

            if excl:
                mask = []
                index_dict = dict(zip(atoms, range(0, len(atoms))))
                for n1 in range(0, n_atoms):
                    m = np.ones(n_atoms)
                    # env_atoms.remove(a1)
                    a1 = atoms[n1]
                    lengths, paths = nx.single_source_dijkstra(a1.mol.G, a1, cutoff=excl)
                    excl_atoms = set(itertools.chain.from_iterable(paths.values()))
                    for a in excl_atoms:
                        if a in index_dict:
                            m[index_dict[a]] = 0
                    mask.append(m)
                mask = np.array(mask)
                d = d * mask

            d.flatten()
            d = d[d != 0.0]

            hist, bin_edges = np.histogram(d, bins=n_bins, range=(0.0, max_dist), normed=False, density=False)
        else:
            hist = np.zeros(n_bins)
        rho = n_atoms / self.box.volume  # number density (N/V)

        for i in range(0, n_bins):
            volBin = (4 / 3.0) * math.pi * (np.power(dr*(i+1), 3) - np.power(dr*i, 3))
            n_ideal = volBin * rho
            val = hist[i] / (n_ideal * n_atoms)
            #val = val / (count * dr)
            rdf[(i+0.5)*dr] = val
        return rdf

    def plot_aa_seq(self):
        bead = np.random.choice(self.beads)
        fig = plt.figure(figsize=(20, 20))
        colors = ["black", "blue", "red", "orange", "green"]
        color_dict = {}
        ax = fig.add_subplot(1,1,1, projection='3d')
        ax.set_title("AA Seq "+bead.type.name, fontsize=40)
        count = 0
        for atom in self.atom_seq_dict[bead]:
            if atom.type not in color_dict:
                color_dict[atom.type] = colors[0]
                colors.remove(colors[0])
            ax.scatter(atom.ref_pos[0], atom.ref_pos[1], atom.ref_pos[2], s=500, marker='o', color=color_dict[atom.type], alpha=0.3)
            ax.text(atom.ref_pos[0], atom.ref_pos[1], atom.ref_pos[2], str(count), fontsize=10)
            count += 1
        #for pos in self.features[self.atom_seq_dict[bead][0]].atom_positions_ref():
        #    ax.scatter(pos[0], pos[1], pos[2], s=200, marker='o', color="yellow", alpha=0.1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlim3d(-0.5, 0.5)
        ax.set_ylim3d(-0.5, 0.5)
        ax.set_zlim3d(-0.5, 0.5)
        plt.show()

    def plot_cg_seq(self):
        mol = np.random.choice(self.mols)
        bead_seq = list(zip(*mol.cg_seq(order=self.order, train=False)))[0]
        fig = plt.figure(figsize=(20, 20))
        colors = ["black", "blue", "red", "orange", "green"]
        color_dict = {}
        ax = fig.add_subplot(1,1,1, projection='3d')
        ax.set_title("CG Seq "+mol.name, fontsize=40)
        count = 0
        center = mol.beads[int(len(mol.beads)/2)].center
        for bead in bead_seq:
            print(bead.index, len(bead.atoms))
            if bead.type not in color_dict:
                color_dict[bead.type] = colors[0]
                colors.remove(colors[0])
            pos = self.box.pbc_diff_vec(bead.center - center)
            ax.scatter(pos[0], pos[1], pos[2], s=1000, marker='o', color=color_dict[bead.type], alpha=0.3)
            ax.text(pos[0], pos[1], pos[2], str(count)+str(bead.index), fontsize=10)
            count += 1
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlim3d(-2.0, 2.0)
        ax.set_ylim3d(-2.0, 2.0)
        ax.set_zlim3d(-2.0, 2.0)
        plt.show()

    def plot_envs(self, bead, train=True, mode="init", only_first=False, cg_kick=0.0):
        if self.heavy_first:
            atom_seq_dict = self.atom_seq_dict_heavy
        else:
            atom_seq_dict = self.atom_seq_dict
        start_atom = atom_seq_dict[bead][0]
        feature = self.features[start_atom]

        cg_feat = self.pad2d(feature.bead_featvec, self.max_beads)
        cg_pos = self.pad2d(feature.bead_positions(cg_kick), self.max_beads)

        if train and mode == "gibbs":
            aa_pos = feature.atom_positions_ref()
        elif train:
            aa_pos = feature.atom_positions_random_mixed(0.4, self.kick)
        else:
            aa_pos = feature.atom_positions()
        aa_pos = self.pad2d(aa_pos, self.max_atoms)

        target_pos, target_type, aa_feat, repl = [], [], [], []
        for atom in atom_seq_dict[bead]:
            feature = self.features[atom]

            t_pos = feature.rot(np.array([atom.ref_pos]))
            target_pos.append(t_pos)
            t_type = np.zeros((1, self.ff.n_atom_chns))
            t_type[0, atom.type.channel] = 1
            target_type.append(t_type)

            if mode == "gibbs" and self.heavy_first:
                print("heavy first")
                atom_featvec = self.pad2d(feature.atom_featvec_gibbs_hf, self.max_atoms)

            elif mode == "gibbs" and not(self.heavy_first):
                atom_featvec = self.pad2d(feature.atom_featvec_gibbs, self.max_atoms)

            else:
                atom_featvec = self.pad2d(feature.atom_featvec_init, self.max_atoms)
            aa_feat.append(atom_featvec)

            r = self.pad1d(feature.repl, self.max_atoms, value=True)
            r = r[np.newaxis, np.newaxis, np.newaxis, :]
            repl.append(r)


        # Padding for recurrent training
        for n in range(0, self.max_seq_len - len(atom_seq_dict[bead])):
            target_pos.append(np.zeros((1, 3)))
            target_type.append(np.zeros((1, self.ff.n_atom_chns)))
            aa_feat.append(np.zeros(aa_feat[-1].shape))
            repl.append(np.ones(repl[-1].shape, dtype=bool))


        for t_pos, aa_feat, repl in zip(target_pos, aa_feat, repl):
            #target_pos, target_type = target
            #atom_pos, atom_featvec, bead_pos, bead_featvec, repl = env
            coords = np.concatenate((aa_pos, cg_pos))
            featvec = np.concatenate((aa_feat, cg_feat))
            _, n_channels = featvec.shape
            fig = plt.figure(figsize=(20,20))
            for c in range(0, n_channels):
                ax = fig.add_subplot(5,6,c+1, projection='3d')
                ax.set_title("Chn. Nr:"+str(c)+" "+self.ff.chn_dict[c], fontsize=4)
                for n in range(0, len(coords)):
                    if featvec[n,c] == 1:
                        ax.scatter(coords[n,0], coords[n,1], coords[n,2], s=5, marker='o', color='black', alpha = 0.5)
                ax.scatter(t_pos[0,0], t_pos[0,1], t_pos[0,2], s=5, marker='o', color='red')
                ax.set_xlim3d(-.8, 0.8)
                ax.set_ylim3d(-.8, 0.8)
                ax.set_zlim3d(-.8, 0.8)
                ax.set_xticks(np.arange(-1, 1, step=0.5))
                ax.set_yticks(np.arange(-1, 1, step=0.5))
                ax.set_zticks(np.arange(-1, 1, step=0.5))
                ax.tick_params(labelsize=6)
                plt.plot([0.0, 0.0], [0.0, 0.0], [-1.0, 1.0])
            plt.show()
            aa_pos = np.where(repl[0,0,0,:, np.newaxis], aa_pos, t_pos)
            if only_first:
                break

    def write_gro_file(self, filename, ref=False):
        elem_dict = {
            "H_AR": "H",
            "H": "H",
            "C_AR": "C",
            "C": "C",
            "B": "B",
            "D": "D",
            "S": "S"
        }
        with open(filename, 'w') as f:
            f.write('{:s}\n'.format('Ala5'))
            f.write('{:5d}\n'.format(self.n_atoms))

            for a in self.atoms:
                if ref:
                    pos = a.ref_pos
                else:
                    pos = a.pos
                f.write('{:5d}{:5s}{:>5s}{:5d}{:8.3f}{:8.3f}{:8.3f}{:8.3f}{:8.3f}{:8.3f}\n'.format(
                    a.mol.index,
                    "sPS",
                    elem_dict[a.type.name]+str(a.mol.atoms.index(a)+1),
                    a.index,
                    pos[0] + a.center[0],
                    pos[1] + a.center[1],
                    pos[2] + a.center[2],
                    0, 0, 0))

            f.write("{:10.5f}{:10.5f}{:10.5f}{:10.5f}{:10.5f}{:10.5f}{:10.5f}{:10.5f}{:10.5f}\n".format(
                self.box.dim[0][0],
                self.box.dim[1][1],
                self.box.dim[2][2],
                self.box.dim[1][0],
                self.box.dim[2][0],
                self.box.dim[0][1],
                self.box.dim[2][1],
                self.box.dim[0][2],
                self.box.dim[1][2]))


class Box():

    def __init__(self, file):

        self.dim = self.get_box_dim(file)
        self.dim_inv= np.linalg.inv(self.dim)
        self.v1 = self.dim[:, 0]
        self.v2 = self.dim[:, 1]
        self.v3 = self.dim[:, 2]
        self.volume = self.get_vol()
        self.center = 0.5*self.v1 + 0.5*self.v2 + 0.5*self.v3

    def get_box_dim(self, file):
        # reads the box dimensions from the last line in the gro file
        f_read = open(file, "r")
        bd = np.array(f_read.readlines()[-1].split(), np.float32)
        f_read.close()
        bd = list(bd)
        for n in range(len(bd), 10):
            bd.append(0.0)
        dim = np.array([[bd[0], bd[5], bd[7]],
                                 [bd[3], bd[1], bd[8]],
                                 [bd[4], bd[6], bd[2]]])
        return dim

    def pbc_old(self, pos):
        if pos[0] > self.v1[0] / 2:
            pos -= self.v1
        elif pos[0] < -self.v1[0] / 2:
            pos += self.v1
        if pos[1] > self.v2[1] / 2:
            pos -= self.v2
        elif pos[1] < -self.v2[1] / 2:
            pos += self.v2
        if pos[2] > self.v3[2] / 2:
            pos -= self.v3
        elif pos[2] < -self.v3[2] / 2:
            pos += self.v3
        return pos

    def pbc_in_box(self, pos):
        f = np.dot(self.dim_inv, pos)
        g = f - np.floor(f)
        new_pos = np.dot(self.dim, g)
        return new_pos

    def pbc_diff_vec(self, diff_vec):
        diff_vec = diff_vec + self.center
        diff_vec = self.pbc_in_box(diff_vec)
        diff_vec = diff_vec - self.center
        return diff_vec

    def pbc_diff_vec_batch(self, diff_vec):
        diff_vec = np.swapaxes(diff_vec, 0, 1)
        diff_vec = diff_vec + self.center[:, np.newaxis]
        diff_vec = self.pbc_in_box(diff_vec)
        diff_vec = diff_vec - self.center[:, np.newaxis]
        diff_vec = np.swapaxes(diff_vec, 0, 1)
        return diff_vec

    def get_vol(self):
        norm1 = np.sqrt(np.sum(np.square(self.v1)))
        norm2 = np.sqrt(np.sum(np.square(self.v2)))
        norm3 = np.sqrt(np.sum(np.square(self.v3)))

        cos1 = np.sum(self.v2 * self.v3) / (norm2 * norm3)
        cos2 = np.sum(self.v1 * self.v3) / (norm1 * norm3)
        cos3 = np.sum(self.v1 * self.v2) / (norm1 * norm2)
        v = norm1*norm2*norm3 * np.sqrt(1-np.square(cos1)-np.square(cos2)-np.square(cos3)+2*np.sqrt(cos1*cos2*cos3))
        return v




#u = Universes(["./data/sPS_t568_small"], "./forcefield/ff.txt", align=True, aug=False, fix_seq=True)
"""
iter = u.traversal_seq(mode ="init")
target_pos, target_type, aa_feat, repl, mask, aa_pos, cg_pos, cg_feat, *_ = next(iter)
u.collection[0].plot_envs(target_pos, aa_feat, repl, aa_pos, cg_pos, cg_feat, only_first=True)
target_pos, target_type, aa_feat, repl, mask, aa_pos, cg_pos, cg_feat, *_ = next(iter)
u.collection[0].plot_envs(target_pos, aa_feat, repl, aa_pos, cg_pos, cg_feat, only_first=True)
target_pos, target_type, aa_feat, repl, mask, aa_pos, cg_pos, cg_feat, *_ = next(iter)
u.collection[0].plot_envs(target_pos, aa_feat, repl, aa_pos, cg_pos, cg_feat, only_first=True)
target_pos, target_type, aa_feat, repl, mask, aa_pos, cg_pos, cg_feat, *_ = next(iter)
u.collection[0].plot_envs(target_pos, aa_feat, repl, aa_pos, cg_pos, cg_feat, only_first=True)
target_pos, target_type, aa_feat, repl, mask, aa_pos, cg_pos, cg_feat, *_ = next(iter)
u.collection[0].plot_envs(target_pos, aa_feat, repl, aa_pos, cg_pos, cg_feat, only_first=True)
target_pos, target_type, aa_feat, repl, mask, aa_pos, cg_pos, cg_feat, *_ = next(iter)
u.collection[0].plot_envs(target_pos, aa_feat, repl, aa_pos, cg_pos, cg_feat, only_first=True)
target_pos, target_type, aa_feat, repl, mask, aa_pos, cg_pos, cg_feat, *_ = next(iter)
u.collection[0].plot_envs(target_pos, aa_feat, repl, aa_pos, cg_pos, cg_feat, only_first=True)
target_pos, target_type, aa_feat, repl, mask, aa_pos, cg_pos, cg_feat, *_ = next(iter)
u.collection[0].plot_envs(target_pos, aa_feat, repl, aa_pos, cg_pos, cg_feat, only_first=True)
target_pos, target_type, aa_feat, repl, mask, aa_pos, cg_pos, cg_feat, *_ = next(iter)
u.collection[0].plot_envs(target_pos, aa_feat, repl, aa_pos, cg_pos, cg_feat, only_first=True)
target_pos, target_type, aa_feat, repl, mask, aa_pos, cg_pos, cg_feat, *_ = next(iter)
u.collection[0].plot_envs(target_pos, aa_feat, repl, aa_pos, cg_pos, cg_feat, only_first=True)
"""
"""
s = timer()
u_bm = deepcopy(u)
print("copy", timer()-s)
s = timer()
iter = u_bm.traversal(train=False, mode="init", batch=True)
print("iter", timer()-s)
s = timer()
count = 0
for batch in iter:
    count += 1
    if count == 100:
        break
print(count, timer()-s)

s = timer()
iter2 = u_bm.traversal(train=False, mode="init", batch=True)
print("iter", timer()-s)
s = timer()
count = 0
for batch in iter2:
    count += 1
    if count == 100:
        break
print(count, timer()-s)
"""

"""

target_pos, target_type, aa_feat, repl, mask, aa_pos, cg_pos, cg_feat = next(iter)

print(target_pos)
print(np.array(target_pos).shape)
print(np.array(target_type).shape)
print(np.array(aa_feat).shape)
print(np.array(repl).shape)
print(np.array(mask).shape)
print(np.array(aa_pos).shape)
print(np.array(cg_pos).shape)
print(np.array(cg_feat).shape)

"""

#u.plot_envs(target_pos, aa_feat, repl, aa_pos, cg_pos, cg_feat)

"""
start = timer()
iter = u.traversal_seq_combined()
target_pos, target_type, aa_feat, repl, mask, aa_pos, cg_pos, cg_feat = next(iter)
#u.plot_envs(target_pos, aa_feat, repl, aa_pos, cg_pos, cg_feat)
target_pos, target_type, aa_feat, repl, mask, aa_pos, cg_pos, cg_feat = next(iter)

#u.plot_envs(target_pos, aa_feat, repl, aa_pos, cg_pos, cg_feat)
target_pos, target_type, aa_feat, repl, mask, aa_pos, cg_pos, cg_feat = next(iter)

#u.plot_envs(target_pos, aa_feat, repl, aa_pos, cg_pos, cg_feat)
target_pos, target_type, aa_feat, repl, mask, aa_pos, cg_pos, cg_feat = next(iter)

#u.plot_envs(target_pos, aa_feat, repl, aa_pos, cg_pos, cg_feat)
target_pos, target_type, aa_feat, repl, mask, aa_pos, cg_pos, cg_feat = next(iter)
#u.plot_envs(target_pos, aa_feat, repl, aa_pos, cg_pos, cg_feat)
end=timer()
print("this took ", end-start)
"""
"""
for target_pos, target_type in t:
    print(target_pos.shape, target_type.shape)
for atom_pos, atom_featvec, bead_pos, bead_featvec, repl in e:
    print(atom_pos.shape, atom_featvec.shape, bead_pos.shape, bead_featvec.shape, repl.shape)
for mask in m:
    print(mask)
print("------")
t, e, m = next(iter)
for target_pos, target_type in t:
    print(target_pos.shape, target_type.shape)
for atom_pos, atom_featvec, bead_pos, bead_featvec, repl in e:
    print(atom_pos.shape, atom_featvec.shape, bead_pos.shape, bead_featvec.shape, repl.shape)
for mask in m:
    print(mask)
print("------")
t, e, m = next(iter)
for target_pos, target_type in t:
    print(target_pos.shape, target_type.shape)
for atom_pos, atom_featvec, bead_pos, bead_featvec, repl in e:
    print(atom_pos.shape, atom_featvec.shape, bead_pos.shape, bead_featvec.shape, repl.shape)
for mask in m:
    print(mask)
print("------")
t, e, m = next(iter)
for target_pos, target_type in t:
    print(target_pos.shape, target_type.shape)
for atom_pos, atom_featvec, bead_pos, bead_featvec, repl in e:
    print(atom_pos.shape, atom_featvec.shape, bead_pos.shape, bead_featvec.shape, repl.shape)
for mask in m:
    print(mask)
print("------")
t, e, m = next(iter)
for target_pos, target_type in t:
    print(target_pos.shape, target_type.shape)
for atom_pos, atom_featvec, bead_pos, bead_featvec, repl in e:
    print(atom_pos.shape, atom_featvec.shape, bead_pos.shape, bead_featvec.shape, repl.shape)
for mask in m:
    print(mask)
print("------")

#seq = u.cg_seq()
#print([[a.index for a in b.seq] for b in list(seq)])
#u.write_gro_file("test.gro")
"""
"""
iter = u.traversal_train()
a,b,c,d,e,f = next(iter)
u.plot_env(a[0], np.concatenate((np.array(c), np.array(e))),np.concatenate((d,f)))
a,b,c,d,e,f = next(iter)
u.plot_env(a[0], np.concatenate((np.array(c), np.array(e))),np.concatenate((d,f)))
a,b,c,d,e,f = next(iter)
u.plot_env(a[0], np.concatenate((np.array(c), np.array(e))),np.concatenate((d,f)))
a,b,c,d,e,f = next(iter)
u.plot_env(a[0], np.concatenate((np.array(c), np.array(e))),np.concatenate((d,f)))
a,b,c,d,e,f = next(iter)
u.plot_env(a[0], np.concatenate((np.array(c), np.array(e))),np.concatenate((d,f)))
a,b,c,d,e,f = next(iter)
u.plot_env(a[0], np.concatenate((np.array(c), np.array(e))),np.concatenate((d,f)))
a,b,c,d,e,f = next(iter)
u.plot_env(a[0], np.concatenate((np.array(c), np.array(e))),np.concatenate((d,f)))
a,b,c,d,e,f = next(iter)
u.plot_env(a[0], np.concatenate((np.array(c), np.array(e))),np.concatenate((d,f)))
a,b,c,d,e,f = next(iter)
a,b,c,d,e,f = next(iter)
a,b,c,d,e,f = next(iter)
a,b,c,d,e,f = next(iter)
a,b,c,d,e,f = next(iter)
a,b,c,d,e,f = next(iter)
"""
"""
print(a)
print(c.shape)
plot_grid(np.sum(voxelize_gauss(c), axis=-1, keepdims=True))

print(e)
print(np.array(e).shape)
print(f.shape)
u.plot_env(a[0], np.concatenate((np.array(c), np.array(e))),np.concatenate((d,f)))

"""
"""
for n in range(0,1000):
    e = next(iter)
    #a1, a2 = e
    #print(a1.index, a1.type.name, a2.index, a2.type.name)
    print(e.index, e.type.name)
"""


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
from dbm.box import *
from dbm.util import read_between
#from utils import *
from copy import deepcopy
#import sys
np.set_printoptions(threshold=np.inf)

class Universe():

    def __init__(self, cfg, path_dict, ff):

        start = timer()

        self.cfg = cfg
        self.aug = int(cfg.getboolean('universe', 'aug'))
        self.align = int(cfg.getboolean('universe', 'align'))
        self.order = cfg.get('universe', 'order')
        self.cutoff = cfg.getfloat('universe', 'cutoff')
        self.kick = cfg.getfloat('universe', 'kick')

        #load forcefield
        self.ff = ff

        cg = md.load(str(path_dict['cg_path']))
        if path_dict['aa_path']:
            aa = md.load(str(path_dict['aa_path']))
        # number of molecules in file
        self.n_mol = cg.topology.n_residues
        # matrix containing box dimensions
        self.box = Box(path_dict['cg_path'])

        # Go through all molecules in cg file and initialize instances of mols and beads
        self.atoms, self.beads, self.mols = [], [], []
        for res in cg.topology.residues:
            self.mols.append(Mol(res.name))

            aa_top_file = path_dict['dir'] / (res.name + "_aa.itp")
            cg_top_file = path_dict['dir'] / (res.name + "_cg.itp")
            map_file = path_dict['dir'] / (res.name + ".map")
            env_file = path_dict['dir'] / (res.name + ".env")

            beads = []
            for bead in res.atoms:
                beads.append(Bead(self.mols[-1],
                                  self.box.move_inside(cg.xyz[0, bead.index]),
                                  self.ff.bead_types[bead.element.symbol]))
                self.mols[-1].add_bead(beads[-1])

            atoms = []
            for line in self.read_between("[map]", "\n", map_file):
                type_name = line.split()[1]
                bead = beads[int(line.split()[2])-1]
                atoms.append(Atom(bead,
                                  self.mols[-1],
                                  bead.center,
                                  self.ff.atom_types[type_name]))


                if path_dict['aa_path']:
                    atoms[-1].ref_pos = self.box.diff_vec(aa.xyz[0, atoms[-1].index] - atoms[-1].center)
                bead.add_atom(atoms[-1])
                self.mols[-1].add_atom(atoms[-1])
            Atom.mol_index = 0

            for line in self.read_between("[bonds]", "[angles]", aa_top_file):
                index1 = int(line.split()[0])-1
                index2 = int(line.split()[1])-1
                bond = self.ff.get_bond([atoms[index1], atoms[index2]])
                if bond:
                    self.mols[-1].add_bond(bond)

            for line in self.read_between("[angles]", "[dihedrals]", aa_top_file):
                index1 = int(line.split()[0])-1
                index2 = int(line.split()[1])-1
                index3 = int(line.split()[2])-1
                angle = self.ff.get_angle([atoms[index1], atoms[index2], atoms[index3]])
                if angle:
                    self.mols[-1].add_angle(angle)

            for line in self.read_between("[dihedrals]", "[exclusions]", aa_top_file):
                index1 = int(line.split()[0])-1
                index2 = int(line.split()[1])-1
                index3 = int(line.split()[2])-1
                index4 = int(line.split()[3])-1
                dih = self.ff.get_dih([atoms[index1], atoms[index2], atoms[index3], atoms[index4]])
                if dih:
                    self.mols[-1].add_dih(dih)

            for line in self.read_between("[exclusions]", "\n", aa_top_file):
                index1 = int(line.split()[0])-1
                index2 = int(line.split()[1])-1
                self.mols[-1].add_excl([atoms[index1], atoms[index2]])

            for line in self.read_between("[bonds]", "[angles]", cg_top_file):
                index1 = int(line.split()[0])-1
                index2 = int(line.split()[1])-1
                self.mols[-1].add_cg_edge([beads[index1], beads[index2]])


            #add atoms and beads to universe
            self.beads += beads
            self.atoms += atoms

            #make Graph for each molecule
            self.mols[-1].make_aa_graph()
            self.mols[-1].make_cg_graph()

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



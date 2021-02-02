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
from dbm.loc_env import *
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

        self.name = path_dict['file_name']

        #parameters
        self.cfg = cfg
        self.aug = int(cfg.getboolean('universe', 'aug'))
        self.align = int(cfg.getboolean('universe', 'align'))
        self.order = cfg.get('universe', 'order')
        self.cutoff_sq = cfg.getfloat('universe', 'cutoff')**2
        self.kick = cfg.getfloat('universe', 'kick')

        #forcefield
        self.ff = ff

        # use mdtraj to load xyz information
        cg = md.load(str(path_dict['cg_path']))
        if path_dict['aa_path']:
            aa = md.load(str(path_dict['aa_path']))
        # number of molecules in file
        self.n_mol = cg.topology.n_residues
        # box dimensions with periodic boundaries
        self.box = Box(path_dict['cg_path'])

        # Go through all molecules in cg file and initialize instances of mols, beads and atoms
        self.atoms, self.beads, self.mols = [], [], []
        for res in cg.topology.residues:
            self.mols.append(Mol(res.name))

            aa_top_file = path_dict['data_dir'] / "aa_top" / (res.name + ".itp")
            cg_top_file = path_dict['data_dir'] / "cg_top" / (res.name + ".itp")
            map_file = path_dict['data_dir'] / "mapping" / (res.name + ".map")
            #env_file = path_dict['dir'] / (res.name + ".env")

            beads = []
            for bead in res.atoms:
                beads.append(Bead(self.mols[-1],
                                  self.box.move_inside(cg.xyz[0, bead.index]),
                                  self.ff.bead_types[bead.element.symbol]))
                self.mols[-1].add_bead(beads[-1])

            atoms = []
            for line in read_between("[map]", "[/map]", map_file):
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

            self.mols[-1].add_aa_top(aa_top_file, self.ff)
            self.mols[-1].add_cg_top(cg_top_file)

            #add atoms and beads to universe
            self.beads += beads
            self.atoms += atoms

            if self.align:
                for line in read_between("[align]", "[/align]", map_file):
                    b_index, fp_index = line.split()
                    if int(b_index) > len(self.mols[-1].beads) or int(fp_index) > len(self.mols[-1].beads):
                        raise Exception('Indices in algn section do not match the molecular structure!')
                    self.mols[-1].beads[int(b_index) - 1].fp = self.mols[-1].beads[int(fp_index) - 1]

            if self.aug:
                for line in read_between("[mult]", "[/mult]", map_file):
                    b_index, m = line.split()
                    if int(b_index) > len(self.mols[-1].beads) or int(m) < 0:
                        raise Exception('Invalid number of multiples!')
                    self.mols[-1].beads[int(b_index) - 1].mult = int(m)

        Atom.index = 0
        Bead.index = 0
        Mol.index = 0
        self.n_atoms = len(self.atoms)

        # generate local envs
        self.loc_envs, self.cg_features, self.aa_seq_heavy, self.aa_seq_hydrogens = {}, {}, {}, {}
        self.tops, self.aa_features = {}, {}
        for mol in self.mols:
            cg_seq, dict_aa_seq_heavy, dict_aa_seq_hydrogens, dict_aa_predecessors = mol.aa_seq(order=self.order,
                                                                                              train=False)
            self.aa_seq_heavy = {**self.aa_seq_heavy, **dict_aa_seq_heavy}
            self.aa_seq_hydrogens = {**self.aa_seq_hydrogens, **dict_aa_seq_hydrogens}
            for bead, _ in cg_seq:
                env_beads = self.get_loc_beads(bead)
                self.loc_envs[bead] = Local_Env(bead, env_beads, self.box)
                self.cg_features[bead] = CG_Feature(self.loc_envs[bead], self.ff)
                for atom in dict_aa_seq_heavy[bead]:
                    self.tops[atom] = Top(atom, self.loc_envs[bead], dict_aa_predecessors[atom], self.ff)
                    self.aa_features[atom] = AA_Feature(self.loc_envs[bead], self.tops[atom])
                for atom in dict_aa_seq_hydrogens[bead]:
                    self.tops[atom] = Top(atom, self.loc_envs[bead], dict_aa_predecessors[atom], self.ff)
                    self.aa_features[atom] = AA_Feature(self.loc_envs[bead], self.tops[atom])

        self.energy = Energy(self.tops, self.box)

        self.kick_atoms()


    def gen_bead_seq(self, train=False):
        bead_seq = []
        mols = self.mols[:]
        np.random.shuffle(mols)
        for mol in mols:
            bead_seq += list(zip(*mol.cg_seq(order=self.order, train=train)))[0]
        return bead_seq

    def kick_atoms(self):
        for a in self.atoms:
            a.pos = np.random.normal(-self.kick, self.kick, 3)

    def get_loc_beads(self, bead):
        centered_positions = np.array([b.center for b in self.beads]) - bead.center
        centered_positions = np.array([self.box.diff_vec(pos) for pos in centered_positions])
        #centered_positions = self.box.pbc_diff_vec_batch(np.array(centered_positions))
        centered_positions_sq = [r[0] * r[0] + r[1] * r[1] + r[2] * r[2] for r in centered_positions]

        indices = np.where(np.array(centered_positions_sq) <= self.cutoff_sq)[0]
        return [self.beads[i] for i in indices]


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

        gen = iter(Generator(self.data, hydrogens=False, gibbs=False, train=False, rand_rot=False, pad_seq=False, ref_pos=False))

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



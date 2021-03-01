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
from dbm.util import read_between

elem_dict = {
    "H_AR": "H",
    "H": "H",
    "C_AR": "C",
    "C": "C"}

class Atom():

    index = 0
    mol_index = 0
    def __init__(self, bead, mol, center, type, ref_pos=None):
        self.index = Atom.index
        Atom.index += 1
        self.mol_index = Atom.mol_index
        Atom.mol_index += 1
        self.bead = bead
        self.mol = mol
        self.center = center
        self.pos = np.zeros(3)
        self.type = type
        self.ref_pos = ref_pos

class Bead():
    index = 1

    def __init__(self, mol, center, type, atoms=None, fp=None, mult=1):
        self.index = Bead.index
        Bead.index += 1
        self.mol = mol
        self.center = center
        self.type = type
        if atoms is None:
            self.atoms = []
        else:
            self.atoms = atoms
        self.loc_beads = None
        self.loc_atoms = None

        self.fp = fp
        self.mult = mult

    def add_atom(self, atom):
        self.atoms.append(atom)


class Mol():

    index = 1
    def __init__(self, name, beads = None, atoms = None):
        self.name = name
        self.index = Mol.index
        Mol.index += 1
        if beads is None:
            self.beads = []
        else:
            self.beads = beads
        if atoms is None:
            self.atoms = []
        else:
            self.atoms = atoms

        self.G = None
        self.G_heavy = None
        self.hydrogens = None

        self.bonds = []
        self.angles = []
        self.dihs = []
        self.excls = []

        self.cg_edges = []

        self.fp = {}

    def add_bead(self, bead):
        self.beads.append(bead)

    def add_atom(self, atom):
        self.atoms.append(atom)

    def add_aa_top(self, top_file, ff):

        for line in read_between("[bonds]", "[angles]", top_file):
            index1 = int(line.split()[0]) - 1
            index2 = int(line.split()[1]) - 1
            bond = ff.make_bond([self.atoms[index1], self.atoms[index2]])
            if bond:
                self.add_bond(bond)

        for line in read_between("[angles]", "[dihedrals]", top_file):
            index1 = int(line.split()[0]) - 1
            index2 = int(line.split()[1]) - 1
            index3 = int(line.split()[2]) - 1
            angle = ff.make_angle([self.atoms[index1], self.atoms[index2], self.atoms[index3]])
            if angle:
                self.add_angle(angle)

        for line in read_between("[dihedrals]", "[exclusions]", top_file):
            index1 = int(line.split()[0]) - 1
            index2 = int(line.split()[1]) - 1
            index3 = int(line.split()[2]) - 1
            index4 = int(line.split()[3]) - 1
            dih = ff.make_dih([self.atoms[index1], self.atoms[index2], self.atoms[index3], self.atoms[index4]])
            if dih:
                self.add_dih(dih)

        for line in read_between("[exclusions]", "\n", top_file):
            index1 = int(line.split()[0]) - 1
            index2 = int(line.split()[1]) - 1
            self.add_excl([self.atoms[index1], self.atoms[index2]])

        self.make_aa_graph()

    def add_cg_top(self, top_file):

        for line in read_between("[bonds]", "[angles]", top_file):
            index1 = int(line.split()[0]) - 1
            index2 = int(line.split()[1]) - 1
            self.add_cg_edge([self.beads[index1], self.beads[index2]])

        self.make_cg_graph()

    def add_bond(self, bond):
        self.bonds.append(bond)

    def add_angle(self, angle):
        self.angles.append(angle)

    def add_dih(self, dih):
        self.dihs.append(dih)

    def add_excl(self, excl):
        self.excls.append(excl)

    def add_cg_edge(self, edge):
        self.cg_edges.append(edge)

    def add_fp(self, bead_ndx, fp_ndx):
        self.fp[self.beads[bead_ndx]] = self.beads[fp_ndx]


    def make_aa_graph(self):
        self.G = nx.Graph()
        self.G.add_nodes_from(self.atoms)
        edges = [bond.atoms for bond in self.bonds]
        self.G.add_edges_from(edges)

        heavy_atoms = [a for a in self.atoms if a.type.mass >= 2.0]
        heavy_edges = [e for e in edges if e[0].type.mass >= 2.0 and e[1].type.mass >= 2.0]
        self.G_heavy = nx.Graph()
        self.G_heavy.add_nodes_from(heavy_atoms)
        self.G_heavy.add_edges_from(heavy_edges)

        self.hydrogens = [a for a in self.atoms if a.type.mass < 2.0]

    def make_cg_graph(self):
        self.G_cg = nx.Graph()
        self.G_cg.add_nodes_from(self.beads)
        self.G_cg.add_edges_from(self.cg_edges)

    def cg_seq(self, order="dfs", train=True):

        #if order == "dfs":
        #    beads = list(nx.dfs_preorder_nodes(self.G_cg))
        #breath first search
        if order == "bfs":
            edges = list(nx.bfs_edges(self.G_cg, np.random.choice(self.beads)))
            beads = [edges[0][0]] + [e[1] for e in edges]
        #random search
        elif order == "random":
            beads = [np.random.choice(self.beads)]
            pool = []
            for n in range(1, len(self.beads)):
                pool += list(nx.neighbors(self.G_cg, beads[-1]))
                pool = list(set(pool))
                next = np.random.choice(pool)
                while next in beads:
                    next = np.random.choice(pool)
                pool.remove(next)
                beads.append(next)
        # depth first search (default)
        else:
            beads = list(nx.dfs_preorder_nodes(self.G_cg))

        # data augmentation for undersampled beads
        seq = []
        for n in range(0, len(beads)):
            if train:
                seq += [(beads[n], beads[:n])]*beads[n].mult
            else:
                seq.append((beads[n], beads[:n]))

        # shuffle sequence if training
        if train:
            np.random.shuffle(seq)

        return seq

    def aa_seq(self, order="dfs", train=True):
        mol_atoms_heavy = [a for a in self.atoms if a.type.mass >= 2.0]
        atom_seq_dict_heavy = {}
        atom_seq_dict_hydrogens = {}
        atom_predecessors_dict = {}
        cg_seq = self.cg_seq(order=order, train=train)
        for bead, predecessor_beads in cg_seq:
            bead_atoms = bead.atoms
            heavy_atoms = [a for a in bead_atoms if a.type.mass >= 2.0]
            hydrogens = [a for a in bead_atoms if a.type.mass < 2.0]
            predecessor_atoms = list(itertools.chain.from_iterable([b.atoms for b in set(predecessor_beads)]))
            predecessor_atoms_heavy = [a for a in predecessor_atoms if a.type.mass >= 2.0]
            predecessor_atoms_hydrogens = [a for a in predecessor_atoms if a.type.mass < 2.0]

            #find start atom
            psble_start_nodes = []
            n_heavy_neighbors = []
            for a in heavy_atoms:
                n_heavy_neighbors.append(len(list(nx.all_neighbors(self.G_heavy, a))))
                for n in nx.all_neighbors(self.G_heavy, a):
                    if n in predecessor_atoms_heavy:
                        psble_start_nodes.append(a)
            if psble_start_nodes:
                #start_atom = np.random.choice(psble_start_nodes)
                #weird bonds in cg sPS... therefore just take first one...
                start_atom = psble_start_nodes[0]
            else:
                start_atom = heavy_atoms[np.array(n_heavy_neighbors).argmin()]
            #else:
            #    start_atom = heavy_atoms[0]

            #sequence through atoms of bead
            if order == "bfs":
                edges = list(nx.bfs_edges(self.G.subgraph(heavy_atoms), start_atom))
                atom_seq = [start_atom] + [e[1] for e in edges]
            elif order == "random":
                atom_seq = [start_atom]
                pool = []
                for n in range(1, len(heavy_atoms)):
                    pool += list(nx.neighbors(self.G.subgraph(heavy_atoms), atom_seq[-1]))
                    pool = list(set(pool))
                    next = np.random.choice(pool)
                    while next in atom_seq:
                        next = np.random.choice(pool)
                    pool.remove(next)
                    atom_seq.append(next)
            else:
                atom_seq = list(nx.dfs_preorder_nodes(self.G.subgraph(heavy_atoms), start_atom))
            #hydrogens = self.hydrogens[:]
            np.random.shuffle(hydrogens)
            #atom_seq = atom_seq + hydrogens

            #atom_seq = []
            for n in range(0, len(atom_seq)):
                atom_predecessors_dict[atom_seq[n]] = predecessor_atoms_heavy + atom_seq[:n]
            for n in range(0, len(hydrogens)):
                atom_predecessors_dict[hydrogens[n]] = mol_atoms_heavy + predecessor_atoms_hydrogens + hydrogens[:n]

            atom_seq_dict_heavy[bead] = atom_seq
            atom_seq_dict_hydrogens[bead] = hydrogens


        return cg_seq, atom_seq_dict_heavy, atom_seq_dict_hydrogens, atom_predecessors_dict



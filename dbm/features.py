import math
import numpy as np
import itertools
from itertools import chain
from collections import Counter
import networkx as nx
from timeit import default_timer as timer
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


class Energy_ndx():

    def __init__(self, loc_env, tops):

        self.loc_env = loc_env
        self.tops = tops

    def bond_ndx(self, key='all'):
        indices = []
        for top in self.tops:
            for bond in top.bonds[key]:
                indices.append(tuple([top.ff.bond_index_dict[bond.type],
                                      self.loc_env.index_dict[bond.atoms[0]],
                                      self.loc_env.index_dict[bond.atoms[1]]]))
        return list(set(indices))

    def angle_ndx(self, key='all'):
        indices = []
        for top in self.tops:
            for angle in top.angles[key]:
                indices.append(tuple([top.ff.angle_index_dict[angle.type],
                                      self.loc_env.index_dict[angle.atoms[0]],
                                      self.loc_env.index_dict[angle.atoms[1]],
                                      self.loc_env.index_dict[angle.atoms[2]]]))
        return list(set(indices))

    def dih_ndx(self, key='all'):
        indices = []
        for top in self.tops:
            for dih in top.dihs[key]:
                indices.append(tuple([top.ff.dih_index_dict[dih.type],
                                      self.loc_env.index_dict[dih.atoms[0]],
                                      self.loc_env.index_dict[dih.atoms[1]],
                                      self.loc_env.index_dict[dih.atoms[2]],
                                      self.loc_env.index_dict[dih.atoms[3]]]))
        return list(set(indices))

    def lj_ndx(self, key='all'):
        indices = []
        for top in self.tops:
            for lj in top.ljs[key]:
                indices.append(tuple([top.ff.lj_index_dict[lj.type],
                                      self.loc_env.index_dict[lj.atoms[0]],
                                      self.loc_env.index_dict[lj.atoms[1]]]))
        return list(set(indices))

class AA_Feature():

    def __init__(self, loc_env, top):

        self.loc_env = loc_env
        self.top = top


        #self.index_dict = dict(zip(self.loc_env.env_atoms, range(0, len(self.loc_env.env_atoms))))

        self.fv_init = self.featvec(key='predecessor')
        self.energy_ndx_init = self.energy_ndx(key='predecessor')
        if top.atom.type.mass >= 2.0:
            self.fv_gibbs = self.featvec(key='heavy')
            self.energy_ndx_gibbs = self.energy_ndx(key='heavy')
        else:
            self.fv_gibbs = self.featvec(key='all')
            self.energy_ndx_gibbs = self.energy_ndx(key='all')

        self.repl = np.ones(len(self.loc_env.atoms), dtype=bool)
        self.repl[self.loc_env.index_dict[self.top.atom]] = False


    def featvec(self, key='all'):
        atom_featvec = np.zeros((len(self.loc_env.atoms), self.top.ff.n_channels))
        for index in range(0, len(self.loc_env.atoms)):
            if self.loc_env.atoms[index].type.channel > 0:
                atom_featvec[index, self.loc_env.atoms[index].type.channel] = 1
        for bond in self.top.bonds[key]:
            indices = self.loc_env.get_indices(bond.atoms)
            atom_featvec[indices, bond.type.channel] = 1
        for angle in self.top.angles[key]:
            indices = self.loc_env.get_indices(angle.atoms)
            atom_featvec[indices, angle.type.channel] = 1
        for dih in self.top.dihs[key]:
            indices = self.loc_env.get_indices(dih.atoms)
            atom_featvec[indices, dih.type.channel] = 1
        for lj in self.top.ljs[key]:
            indices = self.loc_env.get_indices(lj.atoms)
            atom_featvec[indices, lj.type.channel] = 1
        atom_featvec[self.loc_env.index_dict[self.top.atom], :] = 0
        return atom_featvec

    def energy_ndx(self, key='all'):
        d = {'bonds': self.bond_ndx(key),
                 'angles': self.angle_ndx(key),
                 'dihs': self.dih_ndx(key),
                 'ljs': self.lj_ndx(key)}
        return d

    def bond_ndx(self, key='all'):
        indices = []
        for bond in self.top.bonds[key]:
            indices.append(tuple([self.top.ff.bond_index_dict[bond.type],
                            self.loc_env.index_dict[bond.atoms[0]],
                            self.loc_env.index_dict[bond.atoms[1]]]))
        return indices

    def angle_ndx(self, key='all'):
        indices = []
        for angle in self.top.angles[key]:
            indices.append(tuple([self.top.ff.angle_index_dict[angle.type],
                            self.loc_env.index_dict[angle.atoms[0]],
                            self.loc_env.index_dict[angle.atoms[1]],
                            self.loc_env.index_dict[angle.atoms[2]]]))
        return indices

    def dih_ndx(self, key='all'):
        indices = []
        for dih in self.top.dihs[key]:
            indices.append(tuple([self.top.ff.dih_index_dict[dih.type],
                            self.loc_env.index_dict[dih.atoms[0]],
                            self.loc_env.index_dict[dih.atoms[1]],
                            self.loc_env.index_dict[dih.atoms[2]],
                            self.loc_env.index_dict[dih.atoms[3]]]))
        return indices

    def lj_ndx(self, key='all'):
        indices = []
        for lj in self.top.ljs[key]:
            indices.append(tuple([self.top.ff.lj_index_dict[lj.type],
                            self.loc_env.index_dict[lj.atoms[0]],
                            self.loc_env.index_dict[lj.atoms[1]]]))
        return indices


class CG_Feature():

    def __init__(self, loc_env, ff):

        self.loc_env = loc_env
        self.ff = ff

        self.fv = self.featvec()

    def featvec(self):
        bead_featvec = np.zeros((len(self.loc_env.beads), self.ff.n_channels))
        for index in range(0, len(self.loc_env.beads)):
            bead_featvec[index, self.loc_env.beads[index].type.channel] = 1
        bead_featvec[self.loc_env.beads.index(self.loc_env.bead), -1] = 1
        return bead_featvec

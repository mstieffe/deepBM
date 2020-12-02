import math
import numpy as np
from itertools import chain
from collections import Counter

class Env():

    def __init__(self, atom, atoms, beads, ff, box, bonds, angles, dihs, ljs, fp=None):
        self.atom = atom
        self.center = atom.center
        self.beads = beads
        self.atoms = atoms
        self.index_dict = dict(zip(self.atoms,range(0,len(self.atoms))))
        self.atoms_intra = atom.mol.atoms
        self.atoms_inter = list(set(self.atoms)-set(self.atoms_intra))
        self.ff = ff
        self.box = box
        self.bonds = bonds
        self.angles = angles
        self.dihs = dihs
        self.ljs = ljs
        self.n_atoms = len(atoms)
        self.n_channels = self.ff.n_channels
        #self.bond_ndx = self.bond_ndx()
        #self.angle_ndx = self.angle_ndx()
        #self.dih_ndx = self.dih_ndx()
        #self.lj_ndx = self.lj_ndx()

        self.bead_featvec = self.gen_bead_featvec()
        self.atom_featvec_gibbs = self.gen_atom_featvec_gibbs()
        self.atom_featvec_init = None

        #energy indices
        self.bond_ndx_init = None
        self.bond_ndx_gibbs = None
        self.angle_ndx_init = None
        self.angle_ndx_gibbs = None
        self.dih_ndx_init = None
        self.dih_ndx_gibbs = None
        self.lj_ndx_init = None
        self.lj_ndx_gibbs = None

        self.repl = np.ones(len(self.atoms), dtype=bool)
        self.repl[self.atoms.index(atom)] = False

        if fp is None:
            self.rot_mat = np.identity(3)
        else:
            self.rot_mat = self.rot_mat(fp)

    def rot_mat(self, fixpoint):
        #compute rotation matrix to align loc env (aligns fixpoint vector with z-axis)
        v1 = np.array([0.0, 0.0, 1.0])
        v2 = fixpoint

        #rotation axis
        v_rot = np.cross(v1, v2)
        v_rot =  v_rot / np.linalg.norm(v_rot)

        #rotation angle
        cosang = np.dot(v1, v2)
        sinang = np.linalg.norm(np.cross(v1, v2))
        theta = np.arctan2(sinang, cosang)

        #rotation matrix
        a = math.cos(theta / 2.0)
        b, c, d = -v_rot * math.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        rot_mat = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                         [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                         [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

        return rot_mat

    def rot(self, pos):
        return np.dot(pos, self.rot_mat)

    def rot_back(self, pos):
        return np.dot(pos, self.rot_mat.T)

    def get_indices(self, atoms):
        indices = []
        for a in atoms:
            indices.append(self.index_dict[a])
        return indices

    def bead_positions(self):
        positions = [b.center - self.center for b in self.beads]
        #positions = np.dot(np.array(positions), self.rot_mat)
        positions = [self.box.pbc(pos) for pos in positions]
        positions = self.rot(positions)
        return positions

    def gen_bead_featvec(self):
        bead_featvec = np.zeros((len(self.beads), self.n_channels))
        for index in range(0, len(self.beads)):
            bead_featvec[index, self.beads[index].type.channel] = 1
        return bead_featvec

    def atom_positions_bm(self):
        positions = [a.pos + a.center - self.center for a in self.atoms]
        #positions = np.dot(np.array(positions), self.rot_mat)
        positions = [self.box.pbc(pos) for pos in positions]
        positions = self.rot(positions)
        return positions

    def atom_positions_ref(self):
        positions = [a.ref_pos + a.center - self.center for a in self.atoms]
        #positions = np.dot(np.array(positions), self.rot_mat)
        positions = [self.box.pbc(pos) for pos in positions]
        positions = self.rot(positions)
        return positions

    def atom_positions_random_mixed(self, mix_rate, kick):
        positions = []
        for a in self.atoms:
            if a in self.atoms_intra:
                positions.append(a.ref_pos + a.center - self.center)
            else:
                if np.random.uniform() < mix_rate:
                    positions.append(np.random.normal(-kick, kick, 3) + a.center - self.center)
                else:
                    positions.append(a.ref_pos + a.center - self.center)
        #positions = np.dot(np.array(positions), self.rot_mat)
        positions = [self.box.pbc(pos) for pos in positions]
        positions = self.rot(positions)
        return positions

    def get_atom_featvec(self, mode="gibbs", predecessors=None, fix_seq=True):
        if mode == "init":
            if not(np.any(self.atom_featvec_init)) or not(fix_seq):
                self.atom_featvec_init = self.gen_atom_featvec_init(predecessors=predecessors)
            featvec = self.atom_featvec_init
        else:
            if not(np.any(self.atom_featvec_gibbs)):
                self.atom_featvec_gibbs = self.gen_atom_featvec_gibbs()
            featvec = self.atom_featvec_gibbs
        return featvec


    def gen_atom_featvec_gibbs(self):
        atom_featvec = np.zeros((len(self.atoms), self.n_channels))
        for index in range(0, len(self.atoms)):
            atom_featvec[index, self.atoms[index].type.channel] = 1
        for bond in self.bonds:
            #indices = np.isin(self.atoms, bond.atoms).nonzero()[0]
            indices = self.get_indices(bond.atoms)
            atom_featvec[indices, bond.type.channel] = 1
        for angle in self.angles:
            #indices = np.isin(self.atoms, angle.atoms).nonzero()[0]
            indices = self.get_indices(angle.atoms)
            atom_featvec[indices, angle.type.channel] = 1
        for dih in self.dihs:
            #indices = np.isin(self.atoms, dih.atoms).nonzero()[0]
            indices = self.get_indices(dih.atoms)
            atom_featvec[indices, dih.type.channel] = 1
        for lj in self.ljs:
            #indices = np.isin(self.atoms, lj.atoms).nonzero()[0]
            indices = self.get_indices(lj.atoms)
            atom_featvec[indices, lj.type.channel] = 1
        atom_featvec[self.atoms.index(self.atom), :] = 0
        return atom_featvec

    def gen_atom_featvec_init(self, predecessors):
        atom_featvec = np.zeros((len(self.atoms), self.n_channels))
        for index in range(0, len(self.atoms)):
            if self.atoms[index] in predecessors or self.atoms[index] in self.atoms_inter:
                atom_featvec[index, self.atoms[index].type.channel] = 1
        for bond in self.bonds:
            if len(np.isin(predecessors, bond.atoms).nonzero()[0]) == 1:
                indices = np.isin(self.atoms, bond.atoms).nonzero()[0]
                atom_featvec[indices, bond.type.channel] = 1
        for angle in self.angles:
            if len(np.isin(predecessors, angle.atoms).nonzero()[0]) == 2:
                indices = np.isin(self.atoms, angle.atoms).nonzero()[0]
                atom_featvec[indices, angle.type.channel] = 1
        for dih in self.dihs:
            if len(np.isin(predecessors, dih.atoms).nonzero()[0]) == 3:
                indices = np.isin(self.atoms, dih.atoms).nonzero()[0]
                atom_featvec[indices, dih.type.channel] = 1
        for lj in self.ljs:
            if len(np.isin(predecessors + self.atoms_inter, lj.atoms).nonzero()[0]) == 1:
                indices = np.isin(self.atoms, lj.atoms).nonzero()[0]
                atom_featvec[indices, lj.type.channel] = 1
        atom_featvec[self.atoms.index(self.atom), :] = 0
        return atom_featvec


    def get_bond_ndx(self, mode="init", predecessors=None, fix_seq=True):
        if mode == "init":
            if not(np.any(self.bond_ndx_init)) or not(fix_seq):
                self.bond_ndx_init = self.bond_ndx(predecessors=predecessors)
            bond_ndx = self.bond_ndx_init
        else:
            if not(np.any(self.bond_ndx_gibbs)):
                self.bond_ndx_gibbs = self.bond_ndx()
            bond_ndx = self.bond_ndx_gibbs
        return bond_ndx

    def bond_ndx(self, mode="init", predecessors=None):
        indices = []
        for bond in self.bonds:
            #if not(np.any(predecessors)) or len(np.isin(predecessors, bond.atoms).nonzero()[0]) == 1:
            if mode == "gibbs" or check4(set(predecessors), set(bond.atoms), 1):
                indices.append(tuple([self.ff.bond_index_dict[bond.type],
                                self.index_dict[bond.atoms[0]],
                                self.index_dict[bond.atoms[1]]]))

        #return np.array(indices, dtype=np.int32)
        return indices


    def get_angle_ndx(self, mode="init", predecessors=None, fix_seq=True):
        if mode == "init":
            if not(np.any(self.angle_ndx_init)) or not(fix_seq):
                self.angle_ndx_init = self.angle_ndx(predecessors=predecessors)
            angle_ndx = self.angle_ndx_init
        else:
            if not(np.any(self.angle_ndx_gibbs)):
                self.angle_ndx_gibbs = self.angle_ndx()
            angle_ndx = self.angle_ndx_gibbs
        return angle_ndx

    def angle_ndx(self, mode="init", predecessors=None):
        angle_types = self.ff.angle_types
        indices = []
        for angle in self.angles:
            #if not(np.any(predecessors)) or len(np.isin(predecessors, angle.atoms).nonzero()[0]) == 2:
            if mode == "gibbs" or check4(set(predecessors), set(angle.atoms), 2):
                indices.append(tuple([self.ff.angle_index_dict[angle.type],
                                self.index_dict[angle.atoms[0]],
                                self.index_dict[angle.atoms[1]],
                                self.index_dict[angle.atoms[2]]]))
        #return np.array(indices, dtype=np.int32)
        return indices


    def get_dih_ndx(self, mode="init", predecessors=None, fix_seq=True):
        if mode == "init":
            if not(np.any(self.dih_ndx_init)) or not(fix_seq):
                self.dih_ndx_init = self.dih_ndx(predecessors=predecessors)
            dih_ndx = self.dih_ndx_init
        else:
            if not(np.any(self.dih_ndx_gibbs)):
                self.dih_ndx_gibbs = self.dih_ndx()
            dih_ndx = self.dih_ndx_gibbs
        return dih_ndx

    def dih_ndx(self, mode="init", predecessors=None):
        indices = []
        for dih in self.dihs:
            #if not(np.any(predecessors)) or len(np.isin(predecessors, dih.atoms).nonzero()[0]) == 3:
            if mode == "gibbs" or check4(set(predecessors), set(dih.atoms), 3):
                indices.append(tuple([self.ff.dih_index_dict[dih.type],
                                self.index_dict[dih.atoms[0]],
                                self.index_dict[dih.atoms[1]],
                                self.index_dict[dih.atoms[2]],
                                self.index_dict[dih.atoms[3]]]))
        #return np.array(indices, dtype=np.int32)
        return indices


    def get_lj_ndx(self, mode="init", predecessors=None, fix_seq=True):
        if mode == "init":
            if not(np.any(self.lj_ndx_init)) or not(fix_seq):
                self.lj_ndx_init = self.lj_ndx(predecessors=predecessors)
            lj_ndx = self.lj_ndx_init
        else:
            if not(np.any(self.lj_ndx_gibbs)):
                self.lj_ndx_gibbs = self.lj_ndx()
            lj_ndx = self.lj_ndx_gibbs
        return lj_ndx

    def lj_ndx(self, mode="init", predecessors=None):
        indices = []
        if np.any(predecessors):
            #counter = Counter(set(predecessors + self.atoms_inter))
            env_atoms = set(predecessors + self.atoms_inter)
        else:
            env_atoms = set(self.atoms_inter)
        env_atoms = set(self.atoms)
        mode = "gibbs"
        for lj in self.ljs:
            #if not(np.any(predecessors)) or len(np.isin(set(predecessors + self.atoms_inter), set(lj.atoms), assume_unique=True).nonzero()[0]) == 1:
            if mode == "gibbs" or check4(env_atoms, lj.atoms, 2):
                indices.append(tuple([self.ff.lj_index_dict[lj.type],
                                self.index_dict[lj.atoms[0]],
                                self.index_dict[lj.atoms[1]]]))
            #indices.append(tuple([self.ff.lj_index_dict[lj.type],
            #                self.index_dict[lj.atoms[0]],
            #                self.index_dict[lj.atoms[1]]]))
        #return np.array(indices, dtype=np.int32)
        return indices

    def energy(self, ref=False):
        energy = self.bond_energy(ref=ref) + \
                 self.angle_energy(ref=ref) + \
                 self.dih_energy(ref=ref) + \
                 self.lj_energy(ref=ref)
        return energy

    def bond_energy(self, ref=False):
        energy = 0.0
        for bond in self.bonds:
            if ref:
                pos1 = self.box.pbc(bond.atoms[0].ref_pos + bond.atoms[0].center - self.center)
                pos2 = self.box.pbc(bond.atoms[1].ref_pos + bond.atoms[1].center - self.center)
            else:
                pos1 = self.box.pbc(bond.atoms[0].bm_pos + bond.atoms[0].center - self.center)
                pos2 = self.box.pbc(bond.atoms[1].bm_pos + bond.atoms[1].center - self.center)
            energy += self.ff.bond_energy(pos1, pos2, bond.type)
        return energy

    def angle_energy(self, ref=False):
        energy = 0.0
        for angle in self.angles:
            if ref:
                pos1 = self.box.pbc(angle.atoms[0].ref_pos + angle.atoms[0].center - self.center)
                pos2 = self.box.pbc(angle.atoms[1].ref_pos + angle.atoms[1].center - self.center)
                pos3 = self.box.pbc(angle.atoms[2].ref_pos + angle.atoms[2].center - self.center)

            else:
                pos1 = self.box.pbc(angle.atoms[0].bm_pos + angle.atoms[0].center - self.center)
                pos2 = self.box.pbc(angle.atoms[1].bm_pos + angle.atoms[1].center - self.center)
                pos3 = self.box.pbc(angle.atoms[2].bm_pos + angle.atoms[2].center - self.center)
            energy += self.ff.angle_energy(pos1, pos2, pos3, angle.type)
        return energy

    def dih_energy(self, ref=False):
        energy = 0.0
        for dih in self.dihs:
            if ref:
                pos1 = self.box.pbc(dih.atoms[0].ref_pos + dih.atoms[0].center - self.center)
                pos2 = self.box.pbc(dih.atoms[1].ref_pos + dih.atoms[1].center - self.center)
                pos3 = self.box.pbc(dih.atoms[2].ref_pos + dih.atoms[2].center - self.center)
                pos4 = self.box.pbc(dih.atoms[3].ref_pos + dih.atoms[3].center - self.center)
            else:
                pos1 = self.box.pbc(dih.atoms[0].bm_pos + dih.atoms[0].center - self.center)
                pos2 = self.box.pbc(dih.atoms[1].bm_pos + dih.atoms[1].center - self.center)
                pos3 = self.box.pbc(dih.atoms[2].bm_pos + dih.atoms[2].center - self.center)
                pos4 = self.box.pbc(dih.atoms[3].bm_pos + dih.atoms[3].center - self.center)
            energy += self.ff.dih_energy(pos1, pos2, pos3, pos4, dih.type)
        return energy

    def lj_energy(self, ref=False):
        energy = 0.0
        for lj in self.ljs:
            if ref:
                pos1 = self.box.pbc(lj.atoms[0].ref_pos + lj.atoms[0].center - self.center)
                pos2 = self.box.pbc(lj.atoms[1].ref_pos + lj.atoms[1].center - self.center)
            else:
                pos1 = self.box.pbc(lj.atoms[0].bm_pos + lj.atoms[0].center - self.center)
                pos2 = self.box.pbc(lj.atoms[1].bm_pos + lj.atoms[1].center - self.center)
            energy += self.ff.lj_energy(pos1, pos2, lj.type)
        return energy


def check4(set, elems, occ):
    count = 0
    for e in elems:
        if e in set:
            count += 1
    if count == occ:
        return True
    else:
        return False

def check3(counter, elems, occ):
    count = 0
    for e in elems:
        count += counter[e]
    if count == occ:
        return True
    else:
        return False

def check(list, elems, occ):
    counter = Counter(list)
    count = 0
    for e in elems:
        count += counter[e]
    if count == occ:
        return True
    else:
        return False

def check2(list, elems, occ):
    count = 0
    for e in elems:
        count += list.count(e)
    if count == occ:
        return True
    else:
        return False
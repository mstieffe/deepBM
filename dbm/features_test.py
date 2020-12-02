import math
import numpy as np
import itertools
from itertools import chain
from collections import Counter
import networkx as nx
from timeit import default_timer as timer
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


class Feature():

    def __init__(self, atom, predecessors, env_beads, ff, box):

        self.atom = atom
        self.predecessors = predecessors
        self.center = atom.center
        self.env_beads = env_beads
        env_atoms = []
        for b in self.env_beads:
            env_atoms += b.atoms
        self.env_atoms = env_atoms
        self.index_dict = dict(zip(self.env_atoms, range(0,len(self.env_atoms))))
        self.mol = self.atom.mol
        self.bead = self.atom.bead

        self.env_atoms_intra = [a for a in self.env_atoms if a in atom.mol.atoms]
        self.env_atoms_intra_heavy = [a for a in self.env_atoms_intra if a.type.mass >= 2.0]
        self.env_atoms_inter = list(set(self.env_atoms)-set(self.env_atoms_intra))

        self.ff = ff
        #self.n_channels = self.ff.n_channels
        self.box = box

        self.bonds, self.angles, self.dihs, excls = self.get_top(self.atom)
        #self.bonds = self.ff.get_bonds(bond_atoms)
        #self.angles = self.ff.get_angles(angle_atoms)
        #self.dihs = self.ff.get_dihs(dih_atoms)

        lj_atoms_intra = self.get_nonbonded(atom, self.env_atoms_intra, excls)
        self.ljs_intra = self.ff.get_lj(atom, lj_atoms_intra)
        self.ljs_inter = self.ff.get_lj(atom, self.env_atoms_inter)
        self.ljs = self.ljs_intra + self.ljs_inter

        if self.bead.fp is None:
            self.rot_mat = np.identity(3)
        else:
            fp = self.box.pbc_diff_vec(self.bead.fp.center - self.bead.center)
            self.rot_mat = self.rot_mat(fp)

        #feature vectors
        self.bead_featvec = self.get_bead_featvec()
        self.atom_featvec_gibbs = self.get_atom_featvec_gibbs()
        self.atom_featvec_init = self.get_atom_featvec_init()
        if self.atom.type.mass < 2.0:
            self.atom_featvec_gibbs_hf = self.atom_featvec_gibbs
            self.bond_ndx_gibbs = self.get_bond_ndx_gibbs()
            self.angle_ndx_gibbs = self.get_angle_ndx_gibbs()
            self.dih_ndx_gibbs = self.get_dih_ndx_gibbs()
            self.lj_ndx_gibbs = self.get_lj_ndx_gibbs()
        else:
            self.atom_featvec_gibbs_hf = self.get_atom_featvec_gibbs_hf()
            self.bond_ndx_gibbs = self.get_bond_ndx_gibbs_hf()
            self.angle_ndx_gibbs = self.get_angle_ndx_gibbs_hf()
            self.dih_ndx_gibbs = self.get_dih_ndx_gibbs_hf()
            self.lj_ndx_gibbs = self.get_lj_ndx_gibbs_hf()


        #energy indices
        self.bond_ndx_init = self.get_bond_ndx_init()
        #self.bond_ndx_gibbs = self.get_bond_ndx_gibbs()
        self.angle_ndx_init = self.get_angle_ndx_init()
        #self.angle_ndx_gibbs = self.get_angle_ndx_gibbs()
        self.dih_ndx_init = self.get_dih_ndx_init()
        #self.dih_ndx_gibbs = self.get_dih_ndx_gibbs()
        self.lj_ndx_init = self.get_lj_ndx_init()
        #self.lj_ndx_gibbs = self.get_lj_ndx_gibbs()

        #repl vector (used for recurrent training to insert generated atom position in env_atoms for next atom)
        self.repl = np.ones(len(self.env_atoms), dtype=bool)
        self.repl[self.index_dict[self.atom]] = False


    def get_top(self, atom):
        bond_atoms, angle_atoms, dih_atoms, excl_atoms = [], [], [], []
        for bond in self.mol.bonds:
            if atom in bond.atoms:
                bond_atoms.append(bond)

        for angle in self.mol.angles:
            if atom in angle.atoms:
                angle_atoms.append(angle)

        for dih in self.mol.dihs:
            if atom in dih.atoms:
                dih_atoms.append(dih)

        for excl in self.mol.excls:
            if atom in excl:
                excl_atoms.append(excl)
        excl_atoms = list(set(itertools.chain.from_iterable(excl_atoms)))
        if atom in excl_atoms: excl_atoms.remove(atom)

        return bond_atoms, angle_atoms, dih_atoms, excl_atoms

    def get_nonbonded(self, atom, env_atoms, excl_atoms):
        #nexcl_atoms: bonded atoms up to n_excl
        lengths, paths = nx.single_source_dijkstra(atom.mol.G, atom, cutoff=self.ff.n_excl)
        nexcl_atoms = list(set(itertools.chain.from_iterable(paths.values())))
        excl_atoms = nexcl_atoms + excl_atoms
        ljs = list(set(env_atoms) - set(excl_atoms))

        return ljs

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
        positions = [b.center - self.center for b in self.env_beads]
        #positions = np.dot(np.array(positions), self.rot_mat)
        positions = [self.box.pbc_diff_vec(pos) for pos in positions]
        positions = self.rot(positions)
        return positions


    def atom_positions(self):
        positions = [a.pos + a.center - self.center for a in self.env_atoms]
        #positions = np.dot(np.array(positions), self.rot_mat)
        positions = [self.box.pbc_diff_vec(pos) for pos in positions]
        positions = self.rot(positions)
        return positions

    def atom_positions_ref(self):
        positions = [a.ref_pos + a.center - self.center for a in self.env_atoms]
        #positions = np.dot(np.array(positions), self.rot_mat)
        positions = [self.box.pbc_diff_vec(pos) for pos in positions]
        positions = self.rot(positions)
        return positions

    def atom_positions_random_mixed(self, mix_rate, kick):
        positions = []
        for a in self.env_atoms:
            if a in self.env_atoms_intra:
                positions.append(a.ref_pos + a.center - self.center)
            else:
                if np.random.uniform() < mix_rate:
                    positions.append(np.random.normal(-kick, kick, 3) + a.center - self.center)
                else:
                    positions.append(a.ref_pos + a.center - self.center)
        #positions = np.dot(np.array(positions), self.rot_mat)
        positions = [self.box.pbc_diff_vec(pos) for pos in positions]
        positions = self.rot(positions)
        return positions

    """
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
    """

    def get_bead_featvec(self):
        bead_featvec = np.zeros((len(self.env_beads), self.ff.n_channels))
        for index in range(0, len(self.env_beads)):
            bead_featvec[index, self.env_beads[index].type.channel] = 1
        return bead_featvec


    def get_atom_featvec_gibbs(self):
        atom_featvec = np.zeros((len(self.env_atoms), self.ff.n_channels))
        for index in range(0, len(self.env_atoms)):
            atom_featvec[index, self.env_atoms[index].type.channel] = 1
        for bond in self.bonds:
            indices = self.get_indices(bond.atoms)
            atom_featvec[indices, bond.type.channel] = 1
        for angle in self.angles:
            indices = self.get_indices(angle.atoms)
            atom_featvec[indices, angle.type.channel] = 1
        for dih in self.dihs:
            indices = self.get_indices(dih.atoms)
            atom_featvec[indices, dih.type.channel] = 1
        for lj in self.ljs:
            indices = self.get_indices(lj.atoms)
            atom_featvec[indices, lj.type.channel] = 1
        atom_featvec[self.index_dict[self.atom], :] = 0
        return atom_featvec

    def get_atom_featvec_init(self):
        atom_featvec = np.zeros((len(self.env_atoms), self.ff.n_channels))
        for index in range(0, len(self.env_atoms)):
            if self.env_atoms[index] in self.predecessors or self.env_atoms[index] in self.env_atoms_inter:
                atom_featvec[index, self.env_atoms[index].type.channel] = 1
        for bond in self.bonds:
            if len(np.isin(self.predecessors, bond.atoms).nonzero()[0]) == 1:
                indices = self.get_indices(bond.atoms)
                atom_featvec[indices, bond.type.channel] = 1
        for angle in self.angles:
            if len(np.isin(self.predecessors, angle.atoms).nonzero()[0]) == 2:
                indices = self.get_indices(angle.atoms)
                atom_featvec[indices, angle.type.channel] = 1
        for dih in self.dihs:
            if len(np.isin(self.predecessors, dih.atoms).nonzero()[0]) == 3:
                indices = self.get_indices(dih.atoms)
                atom_featvec[indices, dih.type.channel] = 1
        for lj in self.ljs_intra:
            if len(np.isin(self.predecessors, lj.atoms).nonzero()[0]) == 1:
                indices = self.get_indices(lj.atoms)
                atom_featvec[indices, lj.type.channel] = 1
        for lj in self.ljs_inter:
            indices = self.get_indices(lj.atoms)
            atom_featvec[indices, lj.type.channel] = 1
        atom_featvec[self.index_dict[self.atom], :] = 0
        return atom_featvec

    def get_atom_featvec_gibbs_hf(self):
        atom_featvec = np.zeros((len(self.env_atoms), self.ff.n_channels))
        for index in range(0, len(self.env_atoms)):
            if self.env_atoms[index] in self.env_atoms_intra_heavy or self.env_atoms[index] in self.env_atoms_inter:
                atom_featvec[index, self.env_atoms[index].type.channel] = 1
        for bond in self.bonds:
            if len(np.isin(self.env_atoms_intra_heavy, bond.atoms).nonzero()[0]) == 1:
                indices = self.get_indices(bond.atoms)
                atom_featvec[indices, bond.type.channel] = 1
        for angle in self.angles:
            if len(np.isin(self.env_atoms_intra_heavy, angle.atoms).nonzero()[0]) == 2:
                indices = self.get_indices(angle.atoms)
                atom_featvec[indices, angle.type.channel] = 1
        for dih in self.dihs:
            if len(np.isin(self.env_atoms_intra_heavy, dih.atoms).nonzero()[0]) == 3:
                indices = self.get_indices(dih.atoms)
                atom_featvec[indices, dih.type.channel] = 1
        for lj in self.ljs_intra:
            if len(np.isin(self.env_atoms_intra_heavy, lj.atoms).nonzero()[0]) == 1:
                indices = self.get_indices(lj.atoms)
                atom_featvec[indices, lj.type.channel] = 1
        for lj in self.ljs_inter:
            indices = self.get_indices(lj.atoms)
            atom_featvec[indices, lj.type.channel] = 1
        atom_featvec[self.index_dict[self.atom], :] = 0
        return atom_featvec

    def get_bond_ndx_gibbs(self):
        indices = []
        for bond in self.bonds:
            indices.append(tuple([self.ff.bond_index_dict[bond.type],
                            self.index_dict[bond.atoms[0]],
                            self.index_dict[bond.atoms[1]]]))
        return indices

    def get_bond_ndx_gibbs_hf(self):
        indices = []
        for bond in self.bonds:
            if self.check(set(self.env_atoms_intra_heavy), set(bond.atoms), 1):
                indices.append(tuple([self.ff.bond_index_dict[bond.type],
                                self.index_dict[bond.atoms[0]],
                                self.index_dict[bond.atoms[1]]]))
        return indices

    def get_bond_ndx_init(self):
        indices = []
        for bond in self.bonds:
            if self.check(set(self.predecessors), set(bond.atoms), 1):
                indices.append(tuple([self.ff.bond_index_dict[bond.type],
                                self.index_dict[bond.atoms[0]],
                                self.index_dict[bond.atoms[1]]]))
        return indices

    def get_angle_ndx_gibbs(self):
        indices = []
        for angle in self.angles:
            indices.append(tuple([self.ff.angle_index_dict[angle.type],
                            self.index_dict[angle.atoms[0]],
                            self.index_dict[angle.atoms[1]],
                            self.index_dict[angle.atoms[2]]]))
        return indices


    def get_angle_ndx_gibbs_hf(self):
        indices = []
        for angle in self.angles:
            if self.check(set(self.env_atoms_intra_heavy), set(angle.atoms), 2):
                indices.append(tuple([self.ff.angle_index_dict[angle.type],
                                self.index_dict[angle.atoms[0]],
                                self.index_dict[angle.atoms[1]],
                                self.index_dict[angle.atoms[2]]]))

        return indices

    def get_angle_ndx_init(self):
        indices = []
        for angle in self.angles:
            if self.check(set(self.predecessors), set(angle.atoms), 2):
                indices.append(tuple([self.ff.angle_index_dict[angle.type],
                                self.index_dict[angle.atoms[0]],
                                self.index_dict[angle.atoms[1]],
                                self.index_dict[angle.atoms[2]]]))

        return indices

    def get_dih_ndx_gibbs(self):
        indices = []
        for dih in self.dihs:
            indices.append(tuple([self.ff.dih_index_dict[dih.type],
                            self.index_dict[dih.atoms[0]],
                            self.index_dict[dih.atoms[1]],
                            self.index_dict[dih.atoms[2]],
                            self.index_dict[dih.atoms[3]]]))
        return indices

    def get_dih_ndx_gibbs_hf(self):
        indices = []
        for dih in self.dihs:
            if self.check(set(self.env_atoms_intra_heavy), set(dih.atoms), 3):
                indices.append(tuple([self.ff.dih_index_dict[dih.type],
                                self.index_dict[dih.atoms[0]],
                                self.index_dict[dih.atoms[1]],
                                self.index_dict[dih.atoms[2]],
                                self.index_dict[dih.atoms[3]]]))

        return indices

    def get_dih_ndx_init(self):
        indices = []
        for dih in self.dihs:
            if self.check(set(self.predecessors), set(dih.atoms), 3):
                indices.append(tuple([self.ff.dih_index_dict[dih.type],
                                self.index_dict[dih.atoms[0]],
                                self.index_dict[dih.atoms[1]],
                                self.index_dict[dih.atoms[2]],
                                self.index_dict[dih.atoms[3]]]))

        return indices

    def get_lj_ndx_gibbs(self):
        indices = []
        for lj in self.ljs:
            indices.append(tuple([self.ff.lj_index_dict[lj.type],
                            self.index_dict[lj.atoms[0]],
                            self.index_dict[lj.atoms[1]]]))
        return indices

    def get_lj_ndx_gibbs_hf(self):
        indices = []
        for lj in self.ljs_intra:
            if self.check(self.env_atoms_intra_heavy, lj.atoms, 1):
                indices.append(tuple([self.ff.lj_index_dict[lj.type],
                                self.index_dict[lj.atoms[0]],
                                self.index_dict[lj.atoms[1]]]))
        for lj in self.ljs_inter:
            indices.append(tuple([self.ff.lj_index_dict[lj.type],
                            self.index_dict[lj.atoms[0]],
                            self.index_dict[lj.atoms[1]]]))
        return indices

    def get_lj_ndx_init(self):
        indices = []
        for lj in self.ljs_intra:
            if self.check(self.predecessors, lj.atoms, 1):
                indices.append(tuple([self.ff.lj_index_dict[lj.type],
                                self.index_dict[lj.atoms[0]],
                                self.index_dict[lj.atoms[1]]]))
        for lj in self.ljs_inter:
            indices.append(tuple([self.ff.lj_index_dict[lj.type],
                            self.index_dict[lj.atoms[0]],
                            self.index_dict[lj.atoms[1]]]))
        return indices

    def bond_energy(self, ref=False):
        if ref:
            pos1 = [bond.atoms[0].ref_pos + bond.atoms[0].center for bond in self.bonds]
            pos2 = [bond.atoms[1].ref_pos + bond.atoms[1].center for bond in self.bonds]
        else:
            pos1 = [bond.atoms[0].pos + bond.atoms[0].center for bond in self.bonds]
            pos2 = [bond.atoms[1].pos + bond.atoms[1].center for bond in self.bonds]
        equil = [bond.type.equil for bond in self.bonds]
        f_c = [bond.type.force_const for bond in self.bonds]
        dis = self.box.pbc_diff_vec_batch(np.array(pos1) - np.array(pos2))
        dis = np.sqrt(np.sum(np.square(dis), axis=-1))
        bond_energy = self.ff.bond_energy(dis, np.array(equil), np.array(f_c))
        return bond_energy

    def angle_energy(self, ref=False):
        if ref:
            pos1 = [angle.atoms[0].ref_pos + angle.atoms[0].center for angle in self.angles]
            pos2 = [angle.atoms[1].ref_pos + angle.atoms[1].center for angle in self.angles]
            pos3 = [angle.atoms[2].ref_pos + angle.atoms[2].center for angle in self.angles]
        else:
            pos1 = [angle.atoms[0].pos + angle.atoms[0].center for angle in self.angles]
            pos2 = [angle.atoms[1].pos + angle.atoms[1].center for angle in self.angles]
            pos3 = [angle.atoms[2].pos + angle.atoms[2].center for angle in self.angles]
        equil = [angle.type.equil for angle in self.angles]
        f_c = [angle.type.force_const for angle in self.angles]
        if pos1:
            vec1 = self.box.pbc_diff_vec_batch(np.array(pos1) - np.array(pos2))
            vec2 = self.box.pbc_diff_vec_batch(np.array(pos3) - np.array(pos2))
            angle_energy = self.ff.angle_energy(vec1, vec2, equil, f_c)
        else:
            angle_energy = 0.0
        return angle_energy

    def dih_energy(self, ref=False):
        if ref:
            pos1 = [dih.atoms[0].ref_pos + dih.atoms[0].center for dih in self.dihs]
            pos2 = [dih.atoms[1].ref_pos + dih.atoms[1].center for dih in self.dihs]
            pos3 = [dih.atoms[2].ref_pos + dih.atoms[2].center for dih in self.dihs]
            pos4 = [dih.atoms[3].ref_pos + dih.atoms[3].center for dih in self.dihs]
        else:
            pos1 = [dih.atoms[0].pos + dih.atoms[0].center for dih in self.dihs]
            pos2 = [dih.atoms[1].pos + dih.atoms[1].center for dih in self.dihs]
            pos3 = [dih.atoms[2].pos + dih.atoms[2].center for dih in self.dihs]
            pos4 = [dih.atoms[3].pos + dih.atoms[3].center for dih in self.dihs]
        func = [angle.type.func for angle in self.dihs]
        mult = [angle.type.mult for angle in self.dihs]
        equil = [angle.type.equil for angle in self.dihs]
        f_c = [angle.type.force_const for angle in self.dihs]
        if pos1:
            vec1 = self.box.pbc_diff_vec_batch(np.array(pos2) - np.array(pos1))
            vec2 = self.box.pbc_diff_vec_batch(np.array(pos2) - np.array(pos3))
            vec3 = self.box.pbc_diff_vec_batch(np.array(pos4) - np.array(pos3))
            dih_energy = self.ff.dih_energy(vec1, vec2, vec3, func, mult, equil, f_c)
        else:
            dih_energy = 0.0
        return dih_energy

    def lj_energy(self, shift=False, ref=False):
        pos1, pos2, sigma, epsilon = [], [], [], []
        for lj in self.ljs:
            if ref:
                pos1.append(lj.atoms[0].ref_pos + lj.atoms[0].center)
                pos2.append(lj.atoms[1].ref_pos + lj.atoms[1].center)
            else:
                pos1.append(lj.atoms[0].pos + lj.atoms[0].center)
                pos2.append(lj.atoms[1].pos + lj.atoms[1].center)
            sigma.append(lj.type.sigma)
            epsilon.append(lj.type.epsilon)
        dis = self.box.pbc_diff_vec_batch(np.array(pos1) - np.array(pos2))
        dis = np.sqrt(np.sum(np.square(dis), axis=-1))
        energy = self.ff.lj_energy(dis, np.array(sigma), np.array(epsilon), shift=shift)
        return energy

    def lj_energy_intra_inter(self, shift=False, ref=False):
        pos1, pos2, sigma, epsilon = [], [], [], []
        for lj in self.ljs_intra:
            if ref:
                pos1.append(lj.atoms[0].ref_pos + lj.atoms[0].center)
                pos2.append(lj.atoms[1].ref_pos + lj.atoms[1].center)
            else:
                pos1.append(lj.atoms[0].pos + lj.atoms[0].center)
                pos2.append(lj.atoms[1].pos + lj.atoms[1].center)
            sigma.append(lj.type.sigma)
            epsilon.append(lj.type.epsilon)
        if pos1:
            dis = self.box.pbc_diff_vec_batch(np.array(pos1) - np.array(pos2))
            dis = np.sqrt(np.sum(np.square(dis), axis=-1))
            energy_intra = self.ff.lj_energy(dis, np.array(sigma), np.array(epsilon), shift=shift)
        else:
            energy_intra = 0.0

        pos1, pos2, sigma, epsilon = [], [], [], []
        for lj in self.ljs_inter:
            if ref:
                pos1.append(lj.atoms[0].ref_pos + lj.atoms[0].center)
                pos2.append(lj.atoms[1].ref_pos + lj.atoms[1].center)
            else:
                pos1.append(lj.atoms[0].pos + lj.atoms[0].center)
                pos2.append(lj.atoms[1].pos + lj.atoms[1].center)
            sigma.append(lj.type.sigma)
            epsilon.append(lj.type.epsilon)
        if pos1:
            dis = self.box.pbc_diff_vec_batch(np.array(pos1) - np.array(pos2))
            dis = np.sqrt(np.sum(np.square(dis), axis=-1))
            energy_inter = self.ff.lj_energy(dis, np.array(sigma), np.array(epsilon), shift=shift)
        else:
            energy_inter = 0.0
        return energy_intra, energy_inter

    def energy(self, ref=False, shift=False):
        #start = timer()
        bond_energy = self.bond_energy(ref=ref)
        angle_energy = self.angle_energy(ref=ref)
        dih_energy = self.dih_energy(ref=ref)
        lj_energy = self.lj_energy(ref=True, shift=shift)
        #print(timer()-start)
        return bond_energy + angle_energy + dih_energy + lj_energy



    def check(self, set, elems, occ):
        count = 0
        for e in elems:
            if e in set:
                count += 1
        if count == occ:
            return True
        else:
            return False

import itertools
import numpy as np
import networkx as nx
import math

class Local_Env():

    def __init__(self, bead, beads, box):

        self.bead = bead
        #self.atoms = bead.atoms
        self.mol = bead.mol
        self.beads = beads
        atoms = []
        for b in self.beads:
            atoms += b.atoms
        self.atoms = atoms
        self.atoms_intra = [a for a in self.atoms if a in self.mol.atoms]
        self.atoms_intra_heavy = [a for a in self.atoms_intra if a.type.mass >= 2.0]
        self.atoms_inter = list(set(self.atoms)-set(self.atoms_intra))

        self.box = box

        if self.bead.fp is None:
            self.rot_mat = np.identity(3)
        else:
            fp = self.box.diff_vec(self.bead.fp.center - self.bead.center)
            self.rot_mat = self.rot_mat(fp)

        self.index_dict = dict(zip(self.atoms, range(0, len(self.atoms))))


    def get_indices(self, atoms):
        indices = []
        for a in atoms:
            indices.append(self.index_dict[a])
        return indices

    def rot(self, pos):
        return np.dot(pos, self.rot_mat)

    def rot_back(self, pos):
        return np.dot(pos, self.rot_mat.T)

    def bead_positions(self, kick=0.0):
        positions = [b.center - self.bead.center for b in self.beads]
        #positions = np.dot(np.array(positions), self.rot_mat)
        positions = [self.box.diff_vec(pos) for pos in positions]
        positions = self.rot(positions)
        positions = positions + np.random.normal(-kick, kick, positions.shape)
        return positions


    def atom_positions(self):
        positions = [a.pos + a.center - self.bead.center for a in self.atoms]
        #positions = np.dot(np.array(positions), self.rot_mat)
        positions = [self.box.diff_vec(pos) for pos in positions]
        positions = self.rot(positions)
        return positions

    def atom_positions_ref(self):
        positions = [a.ref_pos + a.center - self.bead.center for a in self.atoms]
        #positions = np.dot(np.array(positions), self.rot_mat)
        positions = [self.box.diff_vec(pos) for pos in positions]
        positions = self.rot(positions)
        return positions

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

class Top():

    def __init__(self, atom, loc_env, predecessors, ff):

        self.atom = atom
        self.loc_env = loc_env
        self.predecessors = predecessors
        self.ff = ff

        self.bonds = self.get_bonds()
        self.angles = self.get_angles()
        self.dihs = self.get_dihs()
        self.ljs = self.get_ljs()

    def filter_heavy(self, atoms):
        return [a for a in atoms if a.type.mass >= 2.0]

    def filter_predecessors(self, atoms):
        return [a for a in atoms if a in self.predecessors]

    def get_bonds(self):
        bonds = {'all': [], 'heavy': [], 'predecessor': []}
        for bond in self.atom.mol.bonds:
            if self.atom in bond.atoms:
                bonds['all'].append(bond)
                if len(self.filter_heavy(bond.atoms)) == 2:
                    bonds['heavy'].append(bond)
                if len(self.filter_predecessors(bond.atoms)) == 1:
                    bonds['predecessor'].append(bond)
        return bonds

    def get_angles(self):
        angles = {'all': [], 'heavy': [], 'predecessor': []}
        for angle in self.atom.mol.angles:
            if self.atom in angle.atoms:
                angles['all'].append(angle)
                if len(self.filter_heavy(angle.atoms)) == 3:
                    angles['heavy'].append(angle)
                if len(self.filter_predecessors(angle.atoms)) == 2:
                    angles['predecessor'].append(angle)
        return angles

    def get_dihs(self):
        dihs = {'all': [], 'heavy': [], 'predecessor': []}
        for dih in self.atom.mol.dihs:
            if self.atom in dih.atoms:
                dihs['all'].append(dih)
                if len(self.filter_heavy(dih.atoms)) == 4:
                    dihs['heavy'].append(dih)
                if len(self.filter_predecessors(dih.atoms)) == 3:
                    dihs['predecessor'].append(dih)
        return dihs

    def get_ljs(self):
        ljs = {}
        excl_atoms = []
        for excl in self.atom.mol.excls:
            if self.atom in excl:
                excl_atoms.append(excl)
        excl_atoms = list(set(itertools.chain.from_iterable(excl_atoms)))
        if self.atom in excl_atoms: excl_atoms.remove(self.atom)

        #nexcl_atoms: bonded atoms up to n_excl
        lengths, paths = nx.single_source_dijkstra(self.atom.mol.G, self.atom, cutoff=self.ff.n_excl)
        n_excl_atoms = list(set(itertools.chain.from_iterable(paths.values())))
        excl_atoms = n_excl_atoms + excl_atoms
        lj_atoms_intra = list(set(self.loc_env.atoms_intra) - set(excl_atoms))

        ljs['intra'] = self.ff.make_ljs(self.atom, lj_atoms_intra)
        ljs['intra_heavy'], ljs['intra_predecessor'] = [], []
        for lj in ljs['intra']:
            if len(self.filter_heavy(lj.atoms)) == 2:
                ljs['intra_heavy'].append(lj)
            if len(self.filter_predecessors(lj.atoms)) == 1:
                ljs['intra_predecessor'].append(lj)
        ljs['inter'] = self.ff.make_ljs(self.atom, self.loc_env.atoms_inter)
        ljs['inter_heavy'] = []
        for lj in ljs['inter']:
            if len(self.filter_heavy(lj.atoms)) == 2:
                ljs['inter_heavy'].append(lj)
        ljs['all'] = ljs['intra'] + ljs['inter']
        #ljs['heavy'] = ljs['intra_heavy'] + ljs['inter_heavy']
        ljs['heavy'] = ljs['intra_heavy'] + ljs['inter']

        ljs['predecessor'] = ljs['intra_predecessor'] + ljs['inter']
        #if self.atom.type.mass >= 2.0:
        #    ljs['predecessor'] = ljs['intra_predecessor'] + ljs['inter_heavy']
        #else:
        #    ljs['predecessor'] = ljs['intra_predecessor'] + ljs['inter']


        return ljs


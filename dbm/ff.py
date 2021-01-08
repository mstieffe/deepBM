import math
import numpy as np
from dbm.util import read_between

class Bead_Type():

    def __init__(self, name, channel):
        self.name = name
        self.channel = int(channel)

class Atom_Type():

    def __init__(self, name, channel, mass, charge, sigma, epsilon):
        self.name = name
        self.channel = int(channel)
        self.mass = float(mass)
        self.charge = float(charge)
        self.sigma = float(sigma)
        self.epsilon = float(epsilon)

class Bond_Type():

    def __init__(self, name, channel, func, equil, force_const ):
        self.name = name
        self.channel = int(channel)
        self.func = int(func)
        self.equil = float(equil)
        self.force_const = float(force_const)

class Angle_Type():

    def __init__(self, name, channel, func, equil, force_const):
        self.name = name
        self.channel = int(channel)
        self.func = int(func)
        self.equil = float(equil)*math.pi/180.
        self.force_const = float(force_const)

class Dih_Type():

    def __init__(self, name, channel, func, equil, force_const, mult = 0.0):
        self.name = name
        self.channel = int(channel)
        self.func = int(func)
        self.equil = float(equil)*math.pi/180.
        self.force_const = float(force_const)
        self.mult = float(mult)

class LJ_Type():

    def __init__(self, atom_type1, atom_type2, channel):
        self.name = (atom_type1.name, atom_type2.name)
        self.sigma = 0.5 * (atom_type1.sigma + atom_type2.sigma)
        self.epsilon = math.sqrt(atom_type1.epsilon * atom_type2.epsilon)
        self.channel = int(channel)

class Bond():

    def __init__(self, atoms, type):
        self.atoms = atoms
        self.type = type

class Angle():

    def __init__(self, atoms, type):
        self.atoms = atoms
        self.type = type

class Dih():

    def __init__(self, atoms, type):
        self.atoms = atoms
        self.type = type

class LJ():

    def __init__(self, atoms, type):
        self.atoms = atoms
        self.type = type

class FF():

    def __init__(self, file):
        self.file = file

        #load general information
        for line in read_between("[general]", "[/general]", self.file):
            name, n_excl, n_channels = line.split()
        self.name = name
        self.n_excl = int(n_excl)
        self.n_channels = int(n_channels)

        #load bead types
        self.bead_types = {}
        for line in read_between("[bead_types]", "[/bead_types]", self.file):
            name, channel = line.split()
            self.bead_types[name] = Bead_Type(name, channel)

        #load center bead types
        self.center_bead_types = {}
        for line in read_between("[center_beadtypes]", "[/center_beadtypes]", self.file):
            name, channel = line.split()
            self.center_bead_types[name] = Bead_Type(name, channel)

        #load atom types
        self.atom_types = {}
        for line in read_between("[atom_types]", "[/atom_types]", self.file):
            name, channel, mass, charge, sigma, epsilon = line.split()
            self.atom_types[name] = Atom_Type(name, channel, mass, charge, sigma, epsilon)
        self.n_atom_chns = len(set([atype.channel for atype in self.atom_types.values()]))

        #generate LJ types
        self.lj_types = {}
        for line in read_between("[lj_types]", "[/lj_types]", self.file):
            name1, name2, channel = line.split()
            self.lj_types[(name1, name2)] = LJ_Type(self.atom_types[name1], self.atom_types[name2], channel)
        self.lj_index_dict = dict(zip(self.lj_types.values(), range(0,len(self.lj_types))))


        #load bond types
        self.bond_types = {}
        for line in read_between("[bond_types]", "[/bond_types]", self.file):
            name1, name2, channel, func, equil, force_const = line.split()
            name = (name1, name2)
            self.bond_types[name] = Bond_Type(name, channel, func, equil, force_const)
        self.bond_index_dict = dict(zip(self.bond_types.values(), range(0,len(self.bond_types))))



        #load angle types
        self.angle_types = {}
        for line in read_between("[angle_types]", "[/angle_types]", self.file):
            name1, name2, name3, channel, func, equil, force_const = line.split()
            name = (name1, name2, name3)
            self.angle_types[name] = Angle_Type(name, channel, func, equil, force_const)
        self.angle_index_dict = dict(zip(self.angle_types.values(), range(0,len(self.angle_types))))

        #load dih types
        self.dih_types = {}
        for line in read_between("[dihedral_types]", "[/dihedral_types]", self.file):
            if len(line.split()) == 9:
                name1, name2, name3, name4, channel, func, equil, force_const, mult = line.split()
                name = (name1, name2, name3, name4)
                self.dih_types[name] = Dih_Type(name, channel, func, equil, force_const, mult)
            else:
                name1, name2, name3, name4, channel, func, equil, force_const = line.split()
                name = (name1, name2, name3, name4)
                self.dih_types[name] = Dih_Type(name, channel, func, equil, force_const)
        self.dih_index_dict = dict(zip(self.dih_types.values(),range(0,len(self.dih_types))))

        self.chn_dict = self.make_chn_dict()

    def make_chn_dict(self):
        #dictionary for channel names
        ff_elems = list(self.bead_types.values()) \
                   + list(self.atom_types.values()) \
                   + list(self.center_bead_types.values()) \
                   + list(self.bond_types.values()) \
                   + list(self.angle_types.values()) \
                   + list(self.dih_types.values()) \
                   + list(self.lj_types.values())
        chn_dict = {}
        for o in ff_elems:
            if o.channel in chn_dict:
                chn_dict[o.channel] = chn_dict[o.channel] + "\n" + str(o.name)
            else:
                chn_dict[o.channel] = str(o.name)
        return chn_dict

    def make_ljs(self, atom, non_bonded_atoms):
        ljs = []
        names = [(atom.type.name, non_bonded_atom.type.name) for non_bonded_atom in non_bonded_atoms]
        for name, non_bonded_atom in zip(names, non_bonded_atoms):
            if name in self.lj_types:
                ljs.append(LJ([atom, non_bonded_atom], self.lj_types[name]))
            elif name[::-1] in self.lj_types:
                ljs.append(LJ([atom, non_bonded_atom], self.lj_types[name[::-1]]))
        return ljs

    def make_bond(self, bond_atoms):
        name = tuple([a.type.name for a in bond_atoms])
        if name in self.bond_types:
            return Bond(bond_atoms, self.bond_types[name])
        elif name[::-1] in self.bond_types:
            return Bond(bond_atoms, self.bond_types[name[::-1]])

    def make_angle(self, angle_atoms):
        name = tuple([a.type.name for a in angle_atoms])
        if name in self.angle_types:
            return Angle(angle_atoms, self.angle_types[name])
        elif name[::-1] in self.angle_types:
            return Angle(angle_atoms, self.angle_types[name[::-1]])

    def make_dih(self, dih_atoms):
        name = tuple([a.type.name for a in dih_atoms])
        if name in self.dih_types:
            return Dih(dih_atoms, self.dih_types[name])
        elif name[::-1] in self.dih_types:
            return Dih(dih_atoms, self.dih_types[name[::-1]])


    def bond_params(self):
        params = []
        for bond_type in self.bond_types.values():
            params.append([bond_type.equil, bond_type.force_const])
        params.append([0.0, 0.0]) #dummie for padding..
        return np.array(params)

    def angle_params(self):
        params = []
        for angle_type in self.angle_types.values():
            params.append([angle_type.equil, angle_type.force_const])
        params.append([0.0, 0.0]) #dummie for padding..
        return np.array(params)

    def dih_params(self):
        params = []
        for dih_type in self.dih_types.values():
            params.append([dih_type.equil, dih_type.force_const, dih_type.func, dih_type.mult])
        params.append([0.0, 0.0, 0, 0.0]) #dummie for padding..
        return np.array(params)

    def lj_params(self):
        params = []
        for lj_type in self.lj_types.values():
            params.append([lj_type.sigma, lj_type.epsilon])
        params.append([0.0, 0.0]) #dummie for padding..
        return np.array(params)


class Energy():

    def __init__(self, tops, box, key='all'):
        bonds, angles, dihs, ljs = [], [], [], []
        for top in tops.values():
            bonds += top.bonds['all']
            angles += top.angles['all']
            dihs += top.dihs['all']
            ljs += top.ljs['all']
        self.bonds = list(set(bonds))
        self.angles = list(set(angles))
        self.dihs = list(set(dihs))
        self.ljs = list(set(ljs))

        self.box = box

    def single_bond_pot(self, bond, ref=False):
        if ref:
            pos1 = bond.atoms[0].ref_pos + bond.atoms[0].center
            pos2 = bond.atoms[1].ref_pos + bond.atoms[1].center
        else:
            pos1 = bond.atoms[0].pos + bond.atoms[0].center
            pos2 = bond.atoms[1].pos + bond.atoms[1].center
        dis = self.box.diff_vec(np.array(pos1) - np.array(pos2))
        dis = np.sqrt(np.sum(np.square(dis), axis=-1))
        bond_energy = dis - np.array(bond.type.equil)
        bond_energy = np.square(bond_energy)
        bond_energy = np.array(bond.type.force_const) / 2.0 * bond_energy
        bond_energy = np.sum(bond_energy)
        return bond_energy

    def bond_pot(self, ref=False):
        pos1, pos2, equil, f_c = [], [], [], []
        for bond in self.bonds:
            if ref:
                pos1.append(bond.atoms[0].ref_pos + bond.atoms[0].center)
                pos2.append(bond.atoms[1].ref_pos + bond.atoms[1].center)
            else:
                pos1.append(bond.atoms[0].pos + bond.atoms[0].center)
                pos2.append(bond.atoms[1].pos + bond.atoms[1].center)
            equil.append(bond.type.equil)
            f_c.append(bond.type.force_const)
        dis = self.box.diff_vec_batch(np.array(pos1) - np.array(pos2))
        dis = np.sqrt(np.sum(np.square(dis), axis=-1))
        bond_energy = dis - np.array(equil)
        bond_energy = np.square(bond_energy)
        bond_energy = np.array(f_c) / 2.0 * bond_energy
        bond_energy = np.sum(bond_energy)
        return bond_energy

    def single_angle_pot(self, angle, ref=False):
        if ref:
            pos1 = angle.atoms[0].ref_pos + angle.atoms[0].center
            pos2 = angle.atoms[1].ref_pos + angle.atoms[1].center
            pos3 = angle.atoms[2].ref_pos + angle.atoms[2].center

        else:
            pos1 = angle.atoms[0].pos + angle.atoms[0].center
            pos2 = angle.atoms[1].pos + angle.atoms[1].center
            pos3 = angle.atoms[2].pos + angle.atoms[2].center
        vec1 = self.box.diff_vec(np.array(pos1) - np.array(pos2))
        vec2 = self.box.diff_vec(np.array(pos3) - np.array(pos2))
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
        angle_energy = a - np.array(angle.equil)
        angle_energy = np.square(angle_energy)
        angle_energy = np.array(angle.force_const) / 2.0 * angle_energy
        angle_energy = np.sum(angle_energy)
        return angle_energy


    def angle_pot(self, ref=False):
        pos1, pos2, pos3, equil, f_c = [], [], [], [], []
        for angle in self.angles:
            if ref:
                pos1.append(angle.atoms[0].ref_pos + angle.atoms[0].center)
                pos2.append(angle.atoms[1].ref_pos + angle.atoms[1].center)
                pos3.append(angle.atoms[2].ref_pos + angle.atoms[2].center)
            else:
                pos1.append(angle.atoms[0].pos + angle.atoms[0].center)
                pos2.append(angle.atoms[1].pos + angle.atoms[1].center)
                pos3.append(angle.atoms[2].pos + angle.atoms[2].center)
            equil.append(angle.type.equil)
            f_c.append(angle.type.force_const)
        vec1 = self.box.diff_vec_batch(np.array(pos1) - np.array(pos2))
        vec2 = self.box.diff_vec_batch(np.array(pos3) - np.array(pos2))
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
        angle_energy = a - np.array(equil)
        angle_energy = np.square(angle_energy)
        angle_energy = np.array(f_c) / 2.0 * angle_energy
        angle_energy = np.sum(angle_energy)
        return angle_energy

    def single_dih_pot(self, dih, ref=False):
        if ref:
            pos1 = dih.atoms[0].ref_pos + dih.atoms[0].center
            pos2 = dih.atoms[1].ref_pos + dih.atoms[1].center
            pos3 = dih.atoms[2].ref_pos + dih.atoms[2].center
            pos4 = dih.atoms[3].ref_pos + dih.atoms[3].center
        else:
            pos1 = dih.atoms[0].pos + dih.atoms[0].center
            pos2 = dih.atoms[1].pos + dih.atoms[1].center
            pos3 = dih.atoms[2].pos + dih.atoms[2].center
            pos4 = dih.atoms[3].pos + dih.atoms[3].center

        vec1 = self.box.diff_vec(np.array(pos2) - np.array(pos1))
        vec2 = self.box.diff_vec(np.array(pos2) - np.array(pos3))
        vec3 = self.box.diff_vec(np.array(pos4) - np.array(pos3))
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
        a = np.where(np.array(dih.func) == 1.0, a * np.array(np.array(dih.mult)), a)
        en = a - np.array(dih.equil)
        en = np.where(np.array(dih.func) == 1.0, dih.force_const * (np.cos(en) + 1.0), np.array(dih.force_const) / 2.0 * np.square(en))
        dih_energy = np.sum(en)
        return dih_energy

    def dih_pot(self, ref=False):
        pos1, pos2, pos3, pos4, func, mult, equil, f_c = [], [], [], [], [], [], [], []
        for dih in self.dihs:
            if ref:
                pos1.append(dih.atoms[0].ref_pos + dih.atoms[0].center)
                pos2.append(dih.atoms[1].ref_pos + dih.atoms[1].center)
                pos3.append(dih.atoms[2].ref_pos + dih.atoms[2].center)
                pos4.append(dih.atoms[3].ref_pos + dih.atoms[3].center)
            else:
                pos1.append(dih.atoms[0].pos + dih.atoms[0].center)
                pos2.append(dih.atoms[1].pos + dih.atoms[1].center)
                pos3.append(dih.atoms[2].pos + dih.atoms[2].center)
                pos4.append(dih.atoms[3].pos + dih.atoms[3].center)

            func.append(dih.type.func)
            mult.append(dih.type.mult)
            equil.append(dih.type.equil)
            f_c.append(dih.type.force_const)
        vec1 = self.box.diff_vec_batch(np.array(pos1) - np.array(pos2))
        vec2 = self.box.diff_vec_batch(np.array(pos3) - np.array(pos2))
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
        dih_energy = a - np.array(equil)
        dih_energy = np.square(dih_energy)
        dih_energy = np.array(f_c) / 2.0 * dih_energy
        dih_energy = np.sum(dih_energy)
        return dih_energy


    def lj_pot(self, ljs=None, ref=False, shift=False, cutoff=1.0):
        if not ljs:
            ljs = self.ljs
        pos1, pos2, sigma, epsilon = [], [], [], []
        for lj in ljs:
            if ref:
                pos1.append(lj.atoms[0].ref_pos + lj.atoms[0].center)
                pos2.append(lj.atoms[1].ref_pos + lj.atoms[1].center)
            else:
                pos1.append(lj.atoms[0].pos + lj.atoms[0].center)
                pos2.append(lj.atoms[1].pos + lj.atoms[1].center)
            sigma.append(lj.type.sigma)
            epsilon.append(lj.type.epsilon)
        dis = self.box.diff_vec_batch(np.array(pos1) - np.array(pos2))
        dis = np.sqrt(np.sum(np.square(dis), axis=-1))

        c6_term = np.divide(sigma, dis)
        c6_term = np.power(c6_term, 6)
        c12_term = np.power(c6_term, 2)
        en = np.subtract(c12_term, c6_term)
        en = np.multiply(en, 4 * epsilon)

        if shift:
            c6_term_cut = np.divide(sigma, cutoff)
            c6_term_cut = np.power(c6_term_cut, 6)
            c12_term_cut = np.power(c6_term_cut, 2)
            en_cut = np.subtract(c12_term_cut, c6_term_cut)
            en_cut = np.multiply(en_cut, 4 * np.array(epsilon))

            en = np.where(dis > cutoff, 0.0, en - en_cut)
        en = np.sum(en)
        return en


import math
import numpy as np

class Bead_Type():

    def __init__(self, name, channel):
        #name, channel = line
        self.name = name
        self.channel = int(channel)

class Atom_Type():

    def __init__(self, name, channel, mass, charge, sigma, epsilon):
        #name, channel, mass, charge, sigma, epsilon = line
        self.name = name
        self.channel = int(channel)
        self.mass = float(mass)
        self.charge = float(charge)
        self.sigma = float(sigma)
        self.epsilon = float(epsilon)


class Bond_Type():

    def __init__(self, name, channel, func, equil, force_const ):
        #name1, name2, channel, func, equil, force_const = line
        self.name = name
        self.channel = int(channel)
        self.func = int(func)
        self.equil = float(equil)
        self.force_const = float(force_const)


class Angle_Type():

    def __init__(self, name, channel, func, equil, force_const):
        #name1, name2, name3, channel, func, equil, force_const = line
        self.name = name
        self.channel = int(channel)
        self.func = int(func)
        self.equil = float(equil)*math.pi/180.
        self.force_const = float(force_const)

"""
class Unique_Angle_Type():

    def __init__(self, line):
        mol_name, ndx1, ndx2, ndx3, channel, func, equil, force_const = line
        self.mol_name = str(mol_name)
        self.ndx = {int(ndx1)-1, int(ndx2)-1, int(ndx3)-1}
        self.channel = int(channel)
        self.func = int(func)
        self.equil = float(equil)
        self.force_const = float(force_const)
"""


class Dih_Type():

    def __init__(self, name, channel, func, equil, force_const, mult = 0.0):
        #name1, name2, name3, name4, channel, func, equil, force_const, *_ = line
        self.name = name
        self.channel = int(channel)
        self.func = int(func)
        self.equil = float(equil)*math.pi/180.
        self.force_const = float(force_const)
        self.mult = float(mult)

"""
class Unique_Dih_Type():

    def __init__(self, line):

        mol_name, ndx1, ndx2, ndx3, ndx4, channel, func, equil, force_const, *_ = line
        self.mol_name = str(mol_name)
        self.ndx = {int(ndx1)-1, int(ndx2)-1, int(ndx3)-1, int(ndx4)-1}
        self.channel = int(channel)
        self.func = int(func)
        self.equil = float(equil)
        self.force_const = float(force_const)
        if self.func == 1:
            self.mult = float(line[-1])
        else:
            #self.mult = None
            self.mult = 0.0
"""

class LJ_Type():

    def __init__(self, atom_type1, atom_type2, channel):
        self.name = (atom_type1.name, atom_type2.name)
        self.sigma = 0.5 * (atom_type1.sigma + atom_type2.sigma)
        #self.sigma = math.sqrt(atom_type1.sigma * atom_type2.sigma)
        self.epsilon = math.sqrt(atom_type1.epsilon * atom_type2.epsilon)
        self.channel = int(channel)
        #print("C6 ", 4*self.epsilon*np.power(self.sigma, 6))
        #print("C12 ", 4*self.epsilon*np.power(self.sigma, 12))


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
        for line in self.read_between("[ general ]", "\n"):
            name, n_excl, n_channels = line.split()
        self.name = name
        self.n_excl = int(n_excl)
        self.n_channels = int(n_channels)

        #load bead types
        self.bead_types = {}
        for line in self.read_between("[ beadtypes ]", "\n"):
            name, channel = line.split()
            self.bead_types[name] = Bead_Type(name, channel)

        #load center bead types
        self.center_bead_types = {}
        for line in self.read_between("[ center_beadtypes ]", "\n"):
            name, channel = line.split()
            self.center_bead_types[name] = Bead_Type(name, channel)

        #load atom types
        self.atom_types = {}
        for line in self.read_between("[ atomtypes ]", "\n"):
            name, channel, mass, charge, sigma, epsilon = line.split()
            self.atom_types[name] = Atom_Type(name, channel, mass, charge, sigma, epsilon)
        self.n_atom_chns = len(set([atype.channel for atype in self.atom_types.values()]))

        #generate LJ types
        self.lj_types = {}
        for line in self.read_between("[ ljtypes ]", "\n"):
            name1, name2, channel = line.split()
            self.lj_types[(name1, name2)] = LJ_Type(self.atom_types[name1], self.atom_types[name2], channel)
        self.lj_index_dict = dict(zip(self.lj_types.values(), range(0,len(self.lj_types))))


        #load bond types
        self.bond_types = {}
        for line in self.read_between("[ bondtypes ]", "\n"):
            name1, name2, channel, func, equil, force_const = line.split()
            name = (name1, name2)
            self.bond_types[name] = Bond_Type(name, channel, func, equil, force_const)
        self.bond_index_dict = dict(zip(self.bond_types.values(), range(0,len(self.bond_types))))



        #load angle types
        self.angle_types = {}
        for line in self.read_between("[ angletypes ]", "\n"):
            name1, name2, name3, channel, func, equil, force_const = line.split()
            name = (name1, name2, name3)
            self.angle_types[name] = Angle_Type(name, channel, func, equil, force_const)
        self.angle_index_dict = dict(zip(self.angle_types.values(), range(0,len(self.angle_types))))

        """
        #load unique angle types
        self.unique_angle_types = []
        for line in self.read_between("[ unique angletypes ]", "\n"):
            self.unique_angle_types.append(Unique_Angle_Type(line.split()))
        """

        #load dih types
        self.dih_types = {}
        for line in self.read_between("[ dihedraltypes ]", "\n"):
            if len(line.split()) == 9:
                name1, name2, name3, name4, channel, func, equil, force_const, mult = line.split()
                name = (name1, name2, name3, name4)
                self.dih_types[name] = Dih_Type(name, channel, func, equil, force_const, mult)
            else:
                name1, name2, name3, name4, channel, func, equil, force_const = line.split()
                name = (name1, name2, name3, name4)
                self.dih_types[name] = Dih_Type(name, channel, func, equil, force_const)
        self.dih_index_dict = dict(zip(self.dih_types.values(),range(0,len(self.dih_types))))

        """
        #load unique dih types
        self.unique_dih_types = []
        for line in self.read_between("[ unique dihedraltypes ]", "\n"):
            self.unique_dih_types.append(Unique_Dih_Type(line.split()))
        """
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

    def get_lj(self, atom, non_bonded_atoms):
        ljs = []
        names = [(atom.type.name, non_bonded_atom.type.name) for non_bonded_atom in non_bonded_atoms]
        for name, non_bonded_atom in zip(names, non_bonded_atoms):
            if name in self.lj_types:
                ljs.append(LJ([atom, non_bonded_atom], self.lj_types[name]))
            elif name[::-1] in self.lj_types:
                ljs.append(LJ([atom, non_bonded_atom], self.lj_types[name[::-1]]))
        return ljs

    """
    def get_lj(self, atom, non_bonded_atoms):
        lj_names = [lj_type.name for lj_type in self.lj_types]
        lj = []
        for atom2 in non_bonded_atoms:
            if [atom.type.name, atom2.type.name] in lj_names:
                lj.append(LJ([atom, atom2], self.lj_types[lj_names.index([atom.type.name, atom2.type.name])]))
            elif [atom2.type.name, atom.type.name] in lj_names:
                lj.append(LJ([atom, atom2], self.lj_types[lj_names.index([atom2.type.name, atom.type.name])]))
        return lj
    """

    def get_bond(self, bond_atoms):
        name = tuple([a.type.name for a in bond_atoms])
        if name in self.bond_types:
            return Bond(bond_atoms, self.bond_types[name])
        elif name[::-1] in self.bond_types:
            return Bond(bond_atoms, self.bond_types[name[::-1]])

    def get_angle(self, angle_atoms):
        name = tuple([a.type.name for a in angle_atoms])
        if name in self.angle_types:
            return Angle(angle_atoms, self.angle_types[name])
        elif name[::-1] in self.angle_types:
            return Angle(angle_atoms, self.angle_types[name[::-1]])

    def get_dih(self, dih_atoms):
        name = tuple([a.type.name for a in dih_atoms])
        if name in self.dih_types:
            return Dih(dih_atoms, self.dih_types[name])
        elif name[::-1] in self.dih_types:
            return Dih(dih_atoms, self.dih_types[name[::-1]])

    def get_bonds(self, bond_atoms):
        bonds = []
        names = [tuple([a.type.name for a in atoms]) for atoms in bond_atoms]
        for name, atoms in zip(names, bond_atoms):
            if name in self.bond_types:
                bonds.append(Bond(atoms, self.bond_types[name]))
            elif name[::-1] in self.bond_types:
                bonds.append(Bond(atoms, self.bond_types[name[::-1]]))
        return bonds

    def get_angles(self, angle_atoms):
        angles = []
        names = [tuple([a.type.name for a in atoms]) for atoms in angle_atoms]
        for name, atoms in zip(names, angle_atoms):
            if name in self.angle_types:
                angles.append(Angle(atoms, self.angle_types[name]))
            elif name[::-1] in self.angle_types:
                angles.append(Angle(atoms, self.angle_types[name[::-1]]))
        return angles

    def get_dihs(self, dih_atoms):
        dihs = []
        names = [tuple([a.type.name for a in atoms]) for atoms in dih_atoms]
        for name, atoms in zip(names, dih_atoms):
            if name in self.dih_types:
                dihs.append(Dih(atoms, self.dih_types[name]))
            elif name[::-1] in self.dih_types:
                dihs.append(Dih(atoms, self.dih_types[name[::-1]]))
        return dihs

    """
    def get_bonds(self, suggested_atoms):
        suggested_names = [[a.type.name for a in atoms] for atoms in suggested_atoms]
        bond_names = [bond.name for bond in self.bond_types]
        bonds = []
        for name, atoms in zip(suggested_names, suggested_atoms):
            if name in bond_names:
                bonds.append(Bond(atoms, self.bond_types[bond_names.index(name)]))
            elif name[::-1] in bond_names:
                bonds.append(Bond(atoms, self.bond_types[bond_names.index(name[::-1])]))
        return bonds

    def get_angles(self, suggested_atoms):
        mol_name = suggested_atoms[0][0].mol.name
        suggested_names = [[a.type.name for a in atoms] for atoms in suggested_atoms]
        suggested_ndx = [set([a.mol_index for a in atoms]) for atoms in suggested_atoms]
        angle_names = [angle.name for angle in self.angle_types]
        unique_angle_types = [angle_type for angle_type in self.unique_angle_types if mol_name == angle_type.mol_name]
        unique_angle_indices = [angle_type.ndx for angle_type in unique_angle_types]
        angles = []
        for name, atoms in zip(suggested_names, suggested_atoms):
            if name in angle_names:
                angles.append(Angle(atoms, self.angle_types[angle_names.index(name)]))
            elif name[::-1] in angle_names:
                angles.append(Angle(atoms, self.angle_types[angle_names.index(name[::-1])]))
        for indices, atoms in zip(suggested_ndx, suggested_atoms):
            if indices in unique_angle_indices:
                angles.append(Angle(atoms, unique_angle_types[unique_angle_indices.index(indices)]))


        return angles

    def get_dihs(self, suggested_atoms):
        mol_name = suggested_atoms[0][0].mol.name
        suggested_names = [[a.type.name for a in atoms] for atoms in suggested_atoms]
        suggested_ndx = [set([a.mol_index for a in atoms]) for atoms in suggested_atoms]
        dih_names = [dih.name for dih in self.dih_types]
        unique_dih_types = [dih_type for dih_type in self.unique_dih_types if mol_name == dih_type.mol_name]
        unique_dih_indices = [dih_type.ndx for dih_type in unique_dih_types]
        dihs = []
        for name, atoms in zip(suggested_names, suggested_atoms):
            if name in dih_names:
                dihs.append(Dih(atoms, self.dih_types[dih_names.index(name)]))
            elif name[::-1] in dih_names:
                dihs.append(Dih(atoms, self.dih_types[dih_names.index(name[::-1])]))
        for indices, atoms in zip(suggested_ndx, suggested_atoms):
            if indices in unique_dih_indices:
                dihs.append(Dih(atoms, unique_dih_types[unique_dih_indices.index(indices)]))
        return dihs
        """


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

    def bond_energy(self, dis, equil, f_c):

        bond_energy = dis - np.array(equil)
        bond_energy = np.square(bond_energy)
        bond_energy = np.array(f_c) / 2.0 * bond_energy
        bond_energy = np.sum(bond_energy)
        return bond_energy

    def angle_energy(self, vec1, vec2, equil, f_c):

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

    def dih_energy(self, vec1, vec2, vec3, func, mult, equil, f_c):

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
        a = np.where(np.array(func) == 1.0, a * np.array(np.array(mult)), a)
        en = a - np.array(equil)
        en = np.where(np.array(func) == 1.0, f_c * (np.cos(en) + 1.0), np.array(f_c) / 2.0 * np.square(en))
        dih_energy = np.sum(en)
        return dih_energy

    def lj_energy(self, dis, sigma, epsilon, shift=False, cutoff=1.0):

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
            en_cut = np.multiply(en_cut, 4 * epsilon)

            en = np.where(dis > cutoff, 0.0, en - en_cut)
        en = np.sum(en)
        return en


    def read_between(self, start, end):
        #generator to yield line between start and end
        file = open(self.file)
        rec = False
        for line in file:
            if line.startswith(";"):
                continue
            if not rec:
                if line.startswith(start):
                    rec = True
            elif line.startswith(end):
                rec = False
            else:
                yield line
        file.close()

#ff = FF("ff2.txt")
#print(ff.bead_types[0].channel)
#print([d.mult for d in ff.dihs])
#print(ff.angles[2].type)

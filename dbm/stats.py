import numpy as np
import networkx as nx
from pathlib import Path
from scipy.stats import entropy
from dbm.universe1 import *
from dbm.fig import *

class Stats():

    def __init__(self, data, dir=None):

        self.data = data
        if dir:
            self.path = Path(dir)
        else:
            self.path = Path("./stats/")
        self.path.mkdir(exist_ok=True)

    def evaluate(self, subdir=None):
        # evaluate for every folder stored in data
        for name, samples in zip(self.data.folder_dict.keys(), self.data.folder_dict.values()):
            p = self.path / name
            p.mkdir(exist_ok=True)
            if subdir:
                p = p / subdir
                p.mkdir(exist_ok=True)
            #bonds
            bond_fig = Fig(p/"bonds.pdf", len(self.data.ff.bond_types))
            for bond_name in self.data.ff.bond_types.keys():
                bm_dstr = self.bond_dstr(bond_name, samples)
                ref_dstr = self.bond_dstr(bond_name, samples, ref=True)
                plot_dict = {"title": bond_name, "xlabel": "d [nm]", "ylabel": "p"}
                bond_fig.add_plot(bm_dstr, plot_dict, ref_dstr)
            bond_fig.save()
            #angles
            angle_fig = Fig(p/"angles.pdf", len(self.data.ff.angle_types))
            for angle_name in self.data.ff.angle_types.keys():
                bm_dstr = self.angle_dstr(angle_name, samples)
                ref_dstr = self.angle_dstr(angle_name, samples, ref=True)
                plot_dict = {"title": angle_name, "xlabel": "angle [°]", "ylabel": "p"}
                angle_fig.add_plot(bm_dstr, plot_dict, ref_dstr)
            angle_fig.save()
            #dihs
            dih_fig = Fig(p/"dihs.pdf", len(self.data.ff.dih_types))
            for dih_name in self.data.ff.dih_types.keys():
                bm_dstr = self.dih_dstr(dih_name, samples)
                ref_dstr = self.dih_dstr(dih_name, samples, ref=True)
                plot_dict = {"title": dih_name, "xlabel": "dihedral [°]", "ylabel": "p"}
                dih_fig.add_plot(bm_dstr, plot_dict, ref_dstr)
            dih_fig.save()
            #LJ
            lj_fig = Fig(p/"lj.pdf", len(self.data.ff.dih_types))
            bm_lj = self.lj_per_mol_dstr(samples)
            ref_lj = self.lj_per_mol_dstr(samples, ref=True)
            plot_dict = {"title": "LJ", "xlabel": "E [kJ/mol]", "ylabel": "p"}
            lj_fig.add_plot(bm_lj, plot_dict, ref_lj)
            #LJ carbs only
            bm_lj = self.lj_per_mol_dstr(samples, key='heavy')
            ref_lj = self.lj_per_mol_dstr(samples,key='heavy', ref=True)
            plot_dict = {"title": "LJ (carbs)", "xlabel": "E [kJ/mol]", "ylabel": "p"}
            lj_fig.add_plot(bm_lj, plot_dict, ref_lj)
            lj_fig.save()
            #rdf
            rdf_fig = Fig(p/"rdf.pdf", 2)
            bm_rdf = self.rdf(samples)
            ref_rdf = self.rdf(samples, ref=True)
            plot_dict = {"title": "RDF (all)", "xlabel": "r [nm]", "ylabel": "g(r)"}
            rdf_fig.add_plot(bm_rdf, plot_dict, ref_rdf)
            #rdf carbs
            bm_rdf = self.rdf(samples, species=['C', 'C_AR'])
            ref_rdf = self.rdf(samples, species=['C', 'C_AR'], ref=True)
            plot_dict = {"title": "RDF (carbs)", "xlabel": "r [nm]", "ylabel": "g(r)"}
            rdf_fig.add_plot(bm_rdf, plot_dict, ref_rdf)
            rdf_fig.save()


    def make_histo(self, values, n_bins=80, low=0.0, high=0.2):
        if values.any():
            hist, _ = np.histogram(values, bins=n_bins, range=(low, high), normed=False, density=False)
        else:
            hist = np.zeros(n_bins)

        dstr = {}
        dr = float((high-low) / n_bins)
        for i in range(0, n_bins):
            dstr[low + (i + 0.5) * dr] = hist[i]

        return dstr

    def bond_dstr(self, bond_name, samples, n_bins=80, low=0.0, high=0.2, ref=False):
        #computes the dstr of bond lengths for a given bond type over all samples stored in data

        dis = []
        for sample in samples:
            pos1, pos2 = [], []
            for mol in sample.mols:
                if ref:
                    pos1 += [bond.atoms[0].ref_pos + bond.atoms[0].center for bond in mol.bonds if
                             bond.type.name == bond_name]
                    pos2 += [bond.atoms[1].ref_pos + bond.atoms[1].center for bond in mol.bonds if
                             bond.type.name == bond_name]
                else:
                    pos1 += [bond.atoms[0].pos + bond.atoms[0].center for bond in mol.bonds if
                             bond.type.name == bond_name]
                    pos2 += [bond.atoms[1].pos + bond.atoms[1].center for bond in mol.bonds if
                             bond.type.name == bond_name]
            if pos1:
                dis += list(sample.box.pbc_diff_vec_batch(np.array(pos1) - np.array(pos2)))
        dis = np.sqrt(np.sum(np.square(dis), axis=-1))

        dstr = self.make_histo(dis, n_bins=n_bins, low=low, high=high)

        return dstr

    def angle_dstr(self, angle_name, samples, n_bins=80, low=70.0, high=150., ref=False):
        #computes the dstr of angles for a given angle type over all samples stored in data

        vec1, vec2 = [], []
        for sample in samples:
            pos1, pos2, pos3 = [], [], []
            for mol in sample.mols:
                if ref:
                    pos1 += [angle.atoms[0].ref_pos + angle.atoms[0].center for angle in mol.angles if
                             angle.type.name == angle_name]
                    pos2 += [angle.atoms[1].ref_pos + angle.atoms[1].center for angle in mol.angles if
                             angle.type.name == angle_name]
                    pos3 += [angle.atoms[2].ref_pos + angle.atoms[2].center for angle in mol.angles if
                             angle.type.name == angle_name]

                else:
                    pos1 += [angle.atoms[0].pos + angle.atoms[0].center for angle in mol.angles if
                             angle.type.name == angle_name]
                    pos2 += [angle.atoms[1].pos + angle.atoms[1].center for angle in mol.angles if
                             angle.type.name == angle_name]
                    pos3 += [angle.atoms[2].pos + angle.atoms[2].center for angle in mol.angles if
                             angle.type.name == angle_name]
            if pos1 != []:
                vec1 += list(sample.box.pbc_diff_vec_batch(np.array(pos1) - np.array(pos2)))
                vec2 += list(sample.box.pbc_diff_vec_batch(np.array(pos3) - np.array(pos2)))

        norm1 = np.square(vec1)
        norm1 = np.sum(norm1, axis=-1)
        norm1 = np.sqrt(norm1)
        norm2 = np.square(vec2)
        norm2 = np.sum(norm2, axis=-1)
        norm2 = np.sqrt(norm2)
        norm = np.multiply(norm1, norm2)
        dot = np.multiply(vec1, vec2)
        dot = np.sum(dot, axis=-1)
        angles = np.clip(np.divide(dot, norm), -1.0, 1.0)
        angles = np.arccos(angles)
        angles = angles*180./math.pi

        dstr = self.make_histo(angles, n_bins=n_bins, low=low, high=high)

        return dstr

    def dih_dstr(self, dih_name, samples, n_bins=80, low=0.0, high=360., ref=False):
        #computes the dstr of angles for a given dih type over all samples stored in data

        plane1, plane2 = [], []
        vec1, vec2, vec3 = [], [], []
        for sample in samples:
            pos1, pos2, pos3, pos4 = [], [], [], []
            for mol in sample.mols:
                if ref:
                    pos1 += [dih.atoms[0].ref_pos + dih.atoms[0].center for dih in mol.dihs if
                             dih.type.name == dih_name]
                    pos2 += [dih.atoms[1].ref_pos + dih.atoms[1].center for dih in mol.dihs if
                             dih.type.name == dih_name]
                    pos3 += [dih.atoms[2].ref_pos + dih.atoms[2].center for dih in mol.dihs if
                             dih.type.name == dih_name]
                    pos4 += [dih.atoms[3].ref_pos + dih.atoms[3].center for dih in mol.dihs if
                             dih.type.name == dih_name]

                else:
                    pos1 += [dih.atoms[0].pos + dih.atoms[0].center for dih in mol.dihs if dih.type.name == dih_name]
                    pos2 += [dih.atoms[1].pos + dih.atoms[1].center for dih in mol.dihs if dih.type.name == dih_name]
                    pos3 += [dih.atoms[2].pos + dih.atoms[2].center for dih in mol.dihs if dih.type.name == dih_name]
                    pos4 += [dih.atoms[3].pos + dih.atoms[3].center for dih in mol.dihs if dih.type.name == dih_name]
            if pos1 != []:
                vec1 = sample.box.pbc_diff_vec_batch(np.array(pos2) - np.array(pos1))
                vec2 = sample.box.pbc_diff_vec_batch(np.array(pos2) - np.array(pos3))
                vec3 = sample.box.pbc_diff_vec_batch(np.array(pos4) - np.array(pos3))
                plane1 += list(np.cross(vec1, vec2))
                plane2 += list(np.cross(vec2, vec3))

            norm1 = np.square(plane1)
            norm1 = np.sum(norm1, axis=-1)
            norm1 = np.sqrt(norm1)
            norm2 = np.square(plane2)
            norm2 = np.sum(norm2, axis=-1)
            norm2 = np.sqrt(norm2)
            norm = np.multiply(norm1, norm2)
            dot = np.multiply(plane1, plane2)
            dot = np.sum(dot, axis=-1)
            angles = np.clip(np.divide(dot, norm), -1.0, 1.0)
            angles = np.arccos(angles)
            angles = angles*180./math.pi

        dstr = self.make_histo(angles, n_bins=n_bins, low=low, high=high)

        return dstr

    def lj_per_mol_dstr(self, samples, key='all', n_bins=80, low=-600, high=400.0, ref=False):
        # computes the dstr of molecule-wise lj energies over all samples stored in data

        energies = []
        for sample in samples:
            for mol in sample.mols:
                ljs = [sample.tops[a].ljs[key] for a in mol.atoms]
                ljs = list(set(itertools.chain.from_iterable(ljs)))
                energy = sample.energy.lj_pot(ljs)
                energies.append(energy)

        dstr = self.make_histo(np.array(energies), n_bins=n_bins, low=low, high=high)

        return dstr


    def rdf(self, samples, n_bins=40, max_dist=1.2, species=None, ref=False, excl=3):
        #computes the rdf over all samples stored in data

        rdf = {}
        n_samples = len(samples)
        dr = float(max_dist / n_bins)

        for sample in samples:

            if species:
                atoms = [a for a in sample.atoms if a.type.name in species]
            else:
                atoms = sample.atoms
            n_atoms = len(atoms)

            if ref:
                x = np.array([sample.box.pbc_in_box(a.ref_pos + a.center) for a in atoms])
            else:
                x = np.array([sample.box.pbc_in_box(a.pos + a.center) for a in atoms])

            if atoms != []:
                d = x[:, np.newaxis, :] - x[np.newaxis, :, :]
                d = np.reshape(d, (n_atoms*n_atoms, 3))
                d = sample.box.pbc_diff_vec_batch(d)
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
            rho = n_atoms / sample.box.volume  # number density (N/V)
            for i in range(0, n_bins):
                volBin = (4 / 3.0) * math.pi * (np.power(dr*(i+1), 3) - np.power(dr*i, 3))
                n_ideal = volBin * rho
                val = hist[i] / (n_ideal * n_atoms)
                #val = val / (count * dr)
                if (i+0.5)*dr in rdf:
                    rdf[(i+0.5)*dr] += val/n_samples
                else:
                    rdf[(i+0.5)*dr] = val/n_samples
        return rdf

    def jsd(self, p, q, base=2.0):
        '''
            Implementation of pairwise `jsd` based on
            https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
        '''
        ## convert to np.array
        p, q = np.asarray(p), np.asarray(q)
        ## normalize p, q to probabilities
        p, q = p / p.sum(), q / q.sum()
        m = 1. / 2 * (p + q)
        return entropy(p, m, base=base) / 2. + entropy(q, m, base=base) / 2.

    def bond_energy(self, samples, ref=False):
        energies = []
        for sample in samples:
            energies.append(sample.energy.bond_pot(ref=ref))
        return energies

    def angle_energy(self, samples, ref=False):
        energies = []
        for sample in samples:
            energies.append(sample.energy.angle_pot(ref=ref))
        return energies

    def dih_energy(self, samples, ref=False):
        energies = []
        for sample in samples:
            energies.append(sample.energy.dih_pot(ref=ref))
        return energies


    def lj_energy(self, samples, ref=False, shift=False, cutoff=1.0):
        energies = []
        for sample in samples:
            energies.append(sample.energy.lj_pot(ref=ref, shift=shift, cutoff=cutoff))
        return energies

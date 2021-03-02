import numpy as np
import networkx as nx
from pathlib import Path
from scipy.stats import entropy
from dbm.universe import *
from dbm.fig import *
from dbm.recurrent_generator import Recurrent_Generator
from itertools import islice
import random
import matplotlib.pyplot as plt

class Stats():

    def __init__(self, data, dir=None):

        self.data = data
        if dir:
            self.path = Path(dir)
        else:
            self.path = Path("./stats/")
        self.path.mkdir(exist_ok=True)

    def evaluate(self, train=False, subdir=None, save_samples=False):
        # evaluate for every folder stored in data
        if train:
            samples_dict = self.data.dict_train
        else:
            samples_dict = self.data.dict_val
        for name, samples in zip(samples_dict.keys(), samples_dict.values()):
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
            lj_fig = Fig(p/"lj.pdf", 2)
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

            if save_samples:
                for sample in samples:
                    sample.write_gro_file(p / (sample.name + ".gro"))


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

    def bond_dstr(self, bond_name, samples, n_bins=80, ref=False):
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
                dis += list(sample.box.diff_vec_batch(np.array(pos1) - np.array(pos2)))
        dis = np.sqrt(np.sum(np.square(dis), axis=-1))

        dstr = self.make_histo(dis, n_bins=n_bins, low=np.min(dis)*0.8, high=np.max(dis)*1.2)

        return dstr

    def angle_dstr(self, angle_name, samples, n_bins=80, ref=False):
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
                vec1 += list(sample.box.diff_vec_batch(np.array(pos1) - np.array(pos2)))
                vec2 += list(sample.box.diff_vec_batch(np.array(pos3) - np.array(pos2)))

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

        dstr = self.make_histo(angles, n_bins=n_bins, low=np.min(angles)-20, high=np.max(angles)+20)

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
                vec1 = sample.box.diff_vec_batch(np.array(pos2) - np.array(pos1))
                vec2 = sample.box.diff_vec_batch(np.array(pos2) - np.array(pos3))
                vec3 = sample.box.diff_vec_batch(np.array(pos4) - np.array(pos3))
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

        dstr = self.make_histo(angles, n_bins=n_bins, low=np.min(angles)-20, high=np.max(angles)+20)

        return dstr

    def lj_per_mol_dstr(self, samples, key='all', n_bins=80, low=-600, high=400.0, ref=False):
        # computes the dstr of molecule-wise lj energies over all samples stored in data

        energies = []
        for sample in samples:
            for mol in sample.mols:
                ljs = [sample.tops[a].ljs[key] for a in mol.atoms]
                ljs = list(set(itertools.chain.from_iterable(ljs)))
                energy = sample.energy.lj_pot(ljs, ref=ref)
                energies.append(energy)
        energies = np.array(energies)

        dstr = self.make_histo(energies, n_bins=n_bins, low=np.min(energies)-50, high=np.max(energies)+50)

        return dstr


    def rdf(self, samples, n_bins=40, species=None, ref=False, excl=3):
        #computes the rdf over all samples stored in data

        rdf = {}
        n_samples = len(samples)
        max_dist = 2*self.data.cfg.getfloat('universe', 'cutoff')
        dr = float(max_dist / n_bins)

        for sample in samples:

            if species:
                atoms = [a for a in sample.atoms if a.type.name in species]
            else:
                atoms = sample.atoms
            n_atoms = len(atoms)

            if ref:
                x = np.array([sample.box.move_inside(a.ref_pos + a.center) for a in atoms])
            else:
                x = np.array([sample.box.move_inside(a.pos + a.center) for a in atoms])

            if atoms != []:
                d = x[:, np.newaxis, :] - x[np.newaxis, :, :]
                d = np.reshape(d, (n_atoms*n_atoms, 3))
                d = sample.box.diff_vec_batch(d)
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

    def plot_envs(self, hydrogens=False, gibbs=False, train=False, rand_rot=False, pad_seq=False, ref_pos=False, width=1.0):

        gen = iter(Recurrent_Generator(self.data, hydrogens=hydrogens, gibbs=gibbs, train=train, rand_rot=rand_rot, pad_seq=pad_seq, ref_pos=ref_pos))
        elems = list(islice(gen, 100))
        d = random.choice(elems)

        for t_pos, aa_feat, repl, a in zip(d['target_pos'], d['aa_feat'], d['repl'], d['atom_seq']):
            print(a.type.name)
            print(a.mol_index)
            coords = np.concatenate((d['aa_pos'], d['cg_pos']))
            featvec = np.concatenate((aa_feat, d['cg_feat']))
            _, n_channels = featvec.shape
            fig = plt.figure(figsize=(20,20))

            for c in range(0, n_channels):
                ax = fig.add_subplot(int(np.ceil(np.sqrt(n_channels))),int(np.ceil(np.sqrt(n_channels))),c+1, projection='3d')
                ax.set_title("Chn. Nr:"+str(c)+" "+self.data.ff.chn_dict[c], fontsize=4)
                for n in range(0, len(coords)):
                    if featvec[n,c] == 1:
                        ax.scatter(coords[n,0], coords[n,1], coords[n,2], s=5, marker='o', color='black', alpha = 0.5)
                ax.scatter(t_pos[0,0], t_pos[0,1], t_pos[0,2], s=5, marker='o', color='red')
                ax.set_xlim3d(-width, width)
                ax.set_ylim3d(-width, width)
                ax.set_zlim3d(-width, width)
                ax.set_xticks(np.arange(-1, 1, step=0.5))
                ax.set_yticks(np.arange(-1, 1, step=0.5))
                ax.set_zticks(np.arange(-1, 1, step=0.5))
                ax.tick_params(labelsize=6)
                plt.plot([0.0, 0.0], [0.0, 0.0], [-1.0, 1.0])
            plt.show()
            #print(repl.shape)
            #print(d['aa_pos'].shape)
            #print(t_pos.shape)
            d['aa_pos'] = np.where(repl[:, np.newaxis], d['aa_pos'], t_pos)


    def plot_aa_seq(self):
        sample = np.random.choice(self.data.samples_train+self.data.samples_val)
        bead = np.random.choice(sample.beads)
        fig = plt.figure(figsize=(20, 20))
        colors = ["black", "blue", "red", "orange", "green"]
        color_dict = {}
        ax = fig.add_subplot(1,1,1, projection='3d')
        ax.set_title("AA Seq "+bead.type.name, fontsize=40)
        count = 0
        for atom in sample.aa_seq_heavy[bead]:
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
        sample = np.random.choice(self.data.samples_train+self.data.samples_val)
        mol = np.random.choice(sample.mols)
        bead_seq = list(zip(*mol.cg_seq(order=sample.order, train=False)))[0]
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
            pos = sample.box.diff_vec(bead.center - center)
            ax.scatter(pos[0], pos[1], pos[2], s=100, marker='o', color=color_dict[bead.type], alpha=0.3)
            ax.text(pos[0], pos[1], pos[2], str(count)+ " id:"+str(bead.index), fontsize=6)
            count += 1
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlim3d(-2.0, 2.0)
        ax.set_ylim3d(-2.0, 2.0)
        ax.set_zlim3d(-2.0, 2.0)
        plt.show()
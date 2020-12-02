import numpy as np
from dbm.universe1 import *

class Stats():

    def __init__(self, data):

        self.data = data

    def lj_dstr(self, species=None, n_bins=80, low=-600, high=400.0, ref=False):

        for sample in self.data.samples:
            for mol in sample.mols:

                if species:
                    atoms = [a for a in mol.atoms if a.type.name in species]
                else:
                    atoms = mol.atoms


    def lj_distribution(self, species=None, n_bins=80, low=-600, high=400.0, ref=False):
        lj_dstr = {}
        lj = []
        for mol in self.mols:
            e = 0.0
            if species:
                atoms = [a for a in mol.atoms if a.type.name in species]
            else:
                atoms = mol.atoms
            for atom in atoms:
                if ref:
                    e_intra, e_inter = self.features[atom].lj_energy_intra_inter(ref=True)
                else:
                    e_intra, e_inter = self.features[atom].lj_energy_intra_inter(ref=False)
                e += e_intra/2.0 + e_inter
            lj.append(e)
        #print(len(pos1))
        if lj != []:
            hist, _ = np.histogram(lj, bins=n_bins, range=(low, high), normed=False, density=False)
        else:
            hist = np.zeros(n_bins)

        dr = float((high-low) / n_bins)
        for i in range(0, n_bins):
            lj_dstr[low + (i + 0.5) * dr] = hist[i]
        return lj_dstr


    def lj_dstr2(self):
        dstr = samples[0].lj_distribution()
        values = list(lj_dstr.values())
        keys = list(lj_dstr.keys())
        if len(samples) > 1:
            for u in samples[1:]:
                lj_dstr = u.lj_distribution()
                values = [v1 + v2 for (v1, v2) in zip(values, lj_dstr.values())]
            values = [v / len(samples) for v in values]
        if len(values) == len(keys):
            ax.plot(keys, values, label="bm", color="red", linewidth=2, linestyle='-')

        if ref and ref_exists:
            lj_dstr = samples[0].lj_distribution(ref=True)
            values_ref = list(lj_dstr.values())
            keys = list(lj_dstr.keys())
            if len(self.samples) > 1:
                for u in samples[1:]:
                    lj_dstr = u.lj_distribution(ref=True)
                    values_ref = [v1 + v2 for (v1, v2) in zip(values_ref, lj_dstr.values())]
                values_ref = [v / len(samples) for v in values_ref]
            if len(values_ref) == len(keys):
                ax.plot(keys, values_ref, label="ref", color="black", linewidth=2, linestyle='-')

            with open(folder + "jsd_" + name + tag + ".txt", 'w') as f:
                jsd = self.jsd(values_ref, values)
                f.write('{} {}\n'.format("lj", jsd))
                if np.isnan(jsd):
                    tot_score += 1.0
                else:
                    tot_score += jsd

            results = np.array([keys, values, values_ref])
        else:
            results = np.array([keys, values])
        np.savetxt(data_folder + "lj_" + name, np.transpose(results, [1, 0]))
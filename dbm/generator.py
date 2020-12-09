import numpy as np


class Generator():

    def __init__(self, data, train=False, hydrogens=False, gibbs=False):

        self.data = data
        self.train = train
        self.hydrogens = hydrogens
        self.gibbs = gibbs

    def pad1d(self, vec, max, value=0):
        vec = np.pad(vec, (0, max - len(vec)), 'constant', constant_values=(0, value))
        return vec

    def pad2d(self, vec, max, value=0):
        vec = np.pad(vec, ((0, max - len(vec)), (0, 0)), 'constant', constant_values=(0, value))
        return vec

    def __next__(self):

        #go through every sample
        for sample in self.data.samples:
            #get sequence of beads to visit
            bead_seq = sample.gen_bead_seq(train=self.train)
            #choose dict for atoms in a givne bead (heavy or hydrogens)
            if self.hydrogens:
                atom_seq_dict = sample.atom_seq_dict_hydrogens
            else:
                atom_seq_dict = sample.atom_seq_dict_heavy
            #visit every bead
            for bead in bead_seq:
                d = {}
                start_atom = atom_seq_dict[bead][0]
                feature = sample.features[start_atom]

                #CG features
                d["cg_feat"] = self.pad2d(feature.bead_featvec, self.data.max_beads)
                d["cg_pos"] = self.pad2d(feature.bead_positions(), self.data.max_beads)

                #env atom positions
                if self.train:
                    d["aa_pos"] = self.pad2d(feature.atom_positions_ref(), self.data.max_atoms)
                else:
                    d["aa_pos"] = self.pad2d(feature.atom_positions(), self.data.max_atoms)

                target_pos, target_type, aa_feat, repl = [], [], [], []
                bonds_ndx, angles_ndx, dihs_ndx, ljs_ndx = [], [], [], []
                for atom in atom_seq_dict[bead]:
                    feature = sample.features[atom]

                    #target position
                    t_pos = feature.rot(np.array([atom.ref_pos]))
                    target_pos.append(t_pos)

                    #target atom type
                    t_type = np.zeros(self.data.ff.n_atom_chns)
                    t_type[atom.type.channel] = 1
                    target_type.append(t_type)

                    if self.gibbs:
                        atom_featvec = self.pad2d(feature.atom_featvec_gibbs_hf, self.data.max_atoms)
                        bonds_ndx += feature.bond_ndx_gibbs
                        angles_ndx += feature.angle_ndx_gibbs
                        dihs_ndx += feature.dih_ndx_gibbs
                        ljs_ndx += feature.lj_ndx_gibbs
                    else:
                        atom_featvec = self.pad2d(feature.atom_featvec_init, self.data.max_atoms)
                        bonds_ndx += feature.bond_ndx_init
                        angles_ndx += feature.angle_ndx_init
                        dihs_ndx += feature.dih_ndx_init
                        ljs_ndx += feature.lj_ndx_init

                    #AA featurevector
                    aa_feat.append(atom_featvec)

                    #replace vector: marks the index of the target atom in "aa_pos" (needed for recurrent training)
                    r = self.pad1d(feature.repl, self.data.max_atoms, value=True)
                    repl.append(r)

                #mask for sequences < max_seq_len
                mask = np.zeros(self.data.max_seq_len)
                mask[:len(atom_seq_dict[bead])] = 1
                d["mask"] = mask

                # Padding for recurrent training
                for n in range(0, self.data.max_seq_len - len(atom_seq_dict[bead])):
                    target_pos.append(np.zeros((1, 3)))
                    target_type.append(target_type[-1])
                    aa_feat.append(np.zeros(aa_feat[-1].shape))
                    repl.append(np.ones(repl[-1].shape, dtype=bool))

                # remove duplicates in energy ndx
                bonds_ndx = list(set(bonds_ndx))
                angles_ndx = list(set(angles_ndx))
                dihs_ndx = list(set(dihs_ndx))
                ljs_ndx = list(set(ljs_ndx))

                # padding for energy terms
                for n in range(0, self.data.max_bonds_pb - len(bonds_ndx)):
                    bonds_ndx.append(tuple([-1, 1, 2]))
                for n in range(0, self.data.max_angles_pb - len(angles_ndx)):
                    angles_ndx.append(tuple([-1, 1, 2, 3]))
                for n in range(0, self.data.max_dihs_pb - len(dihs_ndx)):
                    dihs_ndx.append(tuple([-1, 1, 2, 3, 4]))
                for n in range(0, self.data.max_ljs_pb - len(ljs_ndx)):
                    ljs_ndx.append(tuple([-1, 1, 2]))


                yield atom_seq_dict[bead], target_pos, target_type, aa_feat, repl, mask, aa_pos, cg_pos, cg_feat, bonds_ndx, angles_ndx, dihs_ndx, ljs_ndx
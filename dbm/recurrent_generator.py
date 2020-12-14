import numpy as np
import math

class Generator():

    def __init__(self, data, train=False, hydrogens=False, gibbs=False, rand_rot=False):

        self.data = data
        self.train = train
        self.hydrogens = hydrogens
        self.gibbs = gibbs
        self.rand_rot = rand_rot

    def __iter__(self):

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

                #pad energy terms
                d["bonds_ndx"] = self.pad_energy_ndx(bonds_ndx, self.data.max_bonds_pb)
                d["angles_ndx"] = self.pad_energy_ndx(angles_ndx, self.data.max_angles_pb, tuple([-1, 1, 2, 3]))
                d["dihs_ndx"] = self.pad_energy_ndx(dihs_ndx, self.data.max_dihs_pb, tuple([-1, 1, 2, 3, 4]))
                d["ljs_ndx"] = self.pad_energy_ndx(ljs_ndx, self.data.max_ljs_pb)

                # Padding for recurrent training
                for n in range(0, self.data.max_seq_len - len(atom_seq_dict[bead])):
                    target_pos.append(np.zeros((1, 3)))
                    target_type.append(target_type[-1])
                    aa_feat.append(np.zeros(aa_feat[-1].shape))
                    repl.append(np.ones(repl[-1].shape, dtype=bool))
                d["target_pos"] = target_pos
                d["target_type"] = target_type
                d["aa_feat"] = aa_feat
                d["repl"] = repl

                #mask for sequences < max_seq_len
                mask = np.zeros(self.data.max_seq_len)
                mask[:len(atom_seq_dict[bead])] = 1
                d["mask"] = mask

                if self.rand_rot:
                    rot_mat = self.rand_rot_mat()
                    d["target_pos"] = np.dot(d["target_pos"], rot_mat)
                    d["aa_pos"] = np.dot(d["aa_pos"], rot_mat)
                    d["cg_pos"] = np.dot(d["cg_pos"], rot_mat)

                yield atom_seq_dict[bead], d


    def rand_rot_mat(self):
        #rotation axis
        if self.data.align:
            v_rot = np.array([0.0, 0.0, 1.0])
        else:
            phi = np.random.uniform(0, np.pi * 2)
            costheta = np.random.uniform(-1, 1)
            theta = np.arccos(costheta)
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            v_rot = np.array([x, y, z])

        #rotation angle
        theta = np.random.uniform(0, np.pi * 2)

        #rotation matrix
        a = math.cos(theta / 2.0)
        b, c, d = -v_rot * math.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        rot_mat = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                         [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                         [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

        return rot_mat

    def pad1d(self, vec, max, value=0):
        vec = np.pad(vec, (0, max - len(vec)), 'constant', constant_values=(0, value))
        return vec

    def pad2d(self, vec, max, value=0):
        vec = np.pad(vec, ((0, max - len(vec)), (0, 0)), 'constant', constant_values=(0, value))
        return vec

    def pad_energy_ndx(self, ndx, max, value=tuple([-1, 1, 2])):
        #remove dupicates
        ndx = list(set(ndx))
        #pad
        for n in range(0, max - len(ndx)):
            ndx.append(tuple(value))
        return ndx
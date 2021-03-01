import numpy as np
import math

class Recurrent_Fragment_Generator():

    def __init__(self, data, train=False, n_prev=2, n_next=2, n_rec=5, hydrogens=False, gibbs=False, rand_rot=False, pad_seq=True, ref_pos=True):

        self.data = data
        self.train = train

        self.n_prev = n_prev
        self.n_next = n_next
        self.n_rec = n_rec

        self.hydrogens = hydrogens
        self.gibbs = gibbs
        self.rand_rot = rand_rot
        self.pad_seq = pad_seq
        self.ref_pos = ref_pos

        if train:
            self.samples = self.data.samples_train
        else:
            self.samples = self.data.samples_val

    def __iter__(self):

        #go through every sample
        for sample in self.samples:
            #get sequence of beads to visit
            mols = sample.mols[:]
            np.random.shuffle(mols)

            #bead_seq = sample.gen_bead_seq(train=self.train)
            #choose dict for atoms in a givne bead (heavy or hydrogens)
            if self.hydrogens:
                atom_seq_dict = sample.aa_seq_hydrogens
            else:
                atom_seq_dict = sample.aa_seq_heavy
            #visit every bead
            for mol in mols:
                bead_seq = list(zip(*mol.cg_seq(order=sample.order, train=False)))[0]
                for m in range(0, len(bead_seq), self.n_rec):
                    d = {"cg_feat": [],
                         "cg_chn_feat": [],
                         "cg_pos": [],
                         "aa_feat": [],
                         "aa_chn_feat": [],
                         "aa_pos": [],
                         "bead_type": [],
                         "target_pos": [],
                         "repl_ndx": [],
                         "center_diff": [],
                         "bond_ndx": [],
                         "angle_ndx": [],
                         "dih_ndx": [],
                         "lj_ndx": [],
                         "mask": [],
                         "loc_env": [],
                         "atom_seq": []}
                    rot_mat = self.rand_rot_mat()
                    for n in range(0, self.n_rec):
                        print("DFFFFFFFFFFF", n)
                        print(m, len(bead_seq))
                        if m+n < len(bead_seq):
                            bead = bead_seq[m+n]
                            intra_beads = bead_seq[m + n - self.n_prev:m + n + 1 + self.n_next]
                            print(len(intra_beads))
                            #prev_beads = bead_seq[m+n-self.n_prev:m+n]
                            #next_beads = bead_seq[m+n+1:m+n+1+self.n_next]
                            if self.gibbs:
                                intra_atoms = []
                                for b in bead_seq[m+n-self.n_prev:m+n]+bead_seq[m+n+1:m+n+1+self.n_next]:
                                    #intra_atoms += b.atoms
                                    intra_atoms += atom_seq_dict[b]

                            else:
                                intra_atoms = []
                                for b in bead_seq[m+n-self.n_prev:m+n]:
                                    #intra_atoms += b.atoms
                                    intra_atoms += atom_seq_dict[b]

                            #start_atom = atom_seq_dict[bead][0]
                            cg_f = sample.cg_features[bead]
                            loc_env = sample.loc_envs[bead]

                            #CG features
                            d["cg_feat"].append(np.array(self.pad2d(cg_f.fv, self.data.max['beads_loc_env']), dtype=np.float32))
                            d["cg_chn_feat"].append(np.array(self.pad2d(loc_env.chn_cg_featvec(intra_beads), self.data.max['beads_loc_env']), dtype=np.float32))
                            d["cg_pos"].append(np.array(self.pad2d(loc_env.bead_positions(), self.data.max['beads_loc_env']), dtype=np.float32))

                            #env atom positions
                            if self.ref_pos:
                                d["aa_pos"].append(self.pad2d(loc_env.atom_positions_ref(), self.data.max['atoms_loc_env']))
                            else:
                                d["aa_pos"].append(self.pad2d(loc_env.atom_positions(), self.data.max['atoms_loc_env']))
                            d["aa_chn_feat"].append(np.array(self.pad2d(loc_env.chn_aa_featvec(intra_atoms), self.data.max['atoms_loc_env']), dtype=np.float32))
                            aa_featvec = np.zeros((len(loc_env.atoms), self.data.ff.n_channels))
                            for index in range(0, len(loc_env.atoms)):
                                if loc_env.atoms[index].type.channel >= 0:
                                    aa_featvec[index, loc_env.atoms[index].type.channel] = 1
                            d["aa_feat"].append(np.array(self.pad2d(aa_featvec, self.data.max['atoms_loc_env']), dtype=np.float32))

                            #d["target_pos"].append([loc_env.rot(np.array([a.ref_pos])) for a in bead.atoms])
                            target_pos = [a.ref_pos for a in atom_seq_dict[bead]]
                            d["target_pos"].append(self.pad2d(loc_env.rot(np.array(target_pos)), self.data.max['seq_len']))

                            b_type = np.zeros(self.data.ff.n_bead_chns)
                            b_type[bead.type.index] = 1
                            d["bead_type"].append(b_type)

                            if self.rand_rot:
                                d["target_pos"][-1] = np.dot(d["target_pos"][-1], rot_mat)
                                d["aa_pos"][-1] = np.dot(d["aa_pos"][-1], rot_mat)
                                d["cg_pos"][-1] = np.dot(d["cg_pos"][-1], rot_mat)

                            repl = np.ones(self.data.max['atoms_loc_env'], dtype=bool)
                            if n > 0:
                                for a in atom_seq_dict[bead_seq[m+n-1]]:
                                    repl[loc_env.atoms_index_dict[a]] = False
                            d["repl_ndx"].append(repl)

                            d["center_diff"].append(bead.center - bead_seq[m+n-1].center)

                            bonds_ndx, angles_ndx, dihs_ndx, ljs_ndx = [], [], [], []
                            for atom in atom_seq_dict[bead]:
                                aa_f = sample.aa_features[atom]
                                if self.gibbs:
                                    bonds_ndx += aa_f.energy_ndx_gibbs['bonds']
                                    angles_ndx += aa_f.energy_ndx_gibbs['angles']
                                    dihs_ndx += aa_f.energy_ndx_gibbs['dihs']
                                    ljs_ndx += aa_f.energy_ndx_gibbs['ljs']
                                else:
                                    bonds_ndx += aa_f.energy_ndx_init['bonds']
                                    angles_ndx += aa_f.energy_ndx_init['angles']
                                    dihs_ndx += aa_f.energy_ndx_init['dihs']
                                    ljs_ndx += aa_f.energy_ndx_init['ljs']
                            #pad energy terms
                            d["bonds_ndx"] = self.pad_energy_ndx(bonds_ndx, self.data.max['bonds_per_bead'])
                            d["angles_ndx"] = self.pad_energy_ndx(angles_ndx, self.data.max['angles_per_bead'], tuple([-1, 1, 2, 3]))
                            d["dihs_ndx"] = self.pad_energy_ndx(dihs_ndx, self.data.max['dihs_per_bead'], tuple([-1, 1, 2, 3, 4]))
                            d["ljs_ndx"] = self.pad_energy_ndx(ljs_ndx, self.data.max['ljs_per_bead'])


                           #mask for sequences < max_seq_len
                            d["mask"].append(1.0)


                            d['loc_env'].append(loc_env)
                            d['atom_seq'].append(atom_seq_dict[bead])

                        else:
                            #padding
                            d["target_pos"].append(d["target_pos"][-1])
                            d["bead_type"].append(d["bead_type"][-1])
                            d["center_diff"].append(np.zeros(d["center_diff"][-1].shape))
                            d["aa_pos"].append(d["aa_pos"][-1])
                            d["aa_feat"].append(np.zeros(d["aa_feat"][-1].shape))
                            d["aa_feat_chn"].append(np.zeros(d["aa_feat_chn"][-1].shape))
                            d["cg_pos"].append(d["cg_pos"][-1])
                            d["cg_feat"].append(np.zeros(d["cg_feat"][-1].shape))
                            d["cg_feat_chn"].append(np.zeros(d["cg_feat_chn"][-1].shape))
                            d["repl_ndx"].append(np.ones(d["repl_ndx"][-1].shape, dtype=bool))
                            d["mask"].append(0.0)



                    d["target_pos"] = np.array(d["target_pos"], dtype=np.float32)
                    d["bead_type"] = np.array(d["bead_type"], dtype=np.float32)
                    d["center_diff"] = np.array(d["center_diff"], dtype=np.float32)
                    d["aa_pos"] = np.array(d["aa_pos"], dtype=np.float32)
                    d["aa_feat"] = np.array(d["aa_feat"], dtype=np.float32)
                    d["aa_feat_chn"] = np.array(d["aa_feat_chn"], dtype=np.float32)
                    d["cg_pos"] = np.array(d["cg_pos"], dtype=np.float32)
                    d["cg_feat"] = np.array(d["cg_feat"], dtype=np.float32)
                    d["cg_feat_chn"] = np.array(d["cg_feat_chn"], dtype=np.float32)

                    d["repl_ndx"] = np.array(d["repl_ndx"], dtype=np.bool)
                    d["mask"] = np.array(d["mask"], dtype=np.float32)

                    d["bonds_ndx"] = np.array(d["bonds_ndx"] ,dtype=np.int64)
                    d["angles_ndx"] = np.array(d["angles_ndx"] ,dtype=np.int64)
                    d["dihs_ndx"] = np.array(d["dihs_ndx"] ,dtype=np.int64)
                    d["ljs_ndx"] = np.array(d["ljs_ndx"] ,dtype=np.int64)

                    yield d

    def all_elems(self):
        g = iter(self)
        elems = []
        for e in g:
            elems.append(e)
        return elems


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

        rot_mat = rot_mat.astype('float32')

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
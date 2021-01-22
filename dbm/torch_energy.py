import torch
from dbm.ff import *
from scipy import constants
import sys

class Energy_torch():

    def __init__(self, ff, device):
        self.ff = ff
        self.device=device
        self.bond_params = torch.tensor(self.ff.bond_params(), dtype=torch.float32, device=device)
        self.angle_params = torch.tensor(self.ff.angle_params(), dtype=torch.float32, device=device)
        self.dih_params = torch.tensor(self.ff.dih_params(), dtype=torch.float32, device=device)
        self.lj_params = torch.tensor(self.ff.lj_params(), dtype=torch.float32, device=device)
        self.atom_mass = torch.tensor([[[atype.mass for atype in self.ff.atom_types.values()]]], dtype=torch.float32, device=device) #(1, 1, n_atomtypes)

        self.one = torch.tensor(1, dtype=torch.int32, device=device)
        self.bond_min_dist = torch.tensor(0.01, dtype=torch.float32, device=device)
        self.lj_min_dist = torch.tensor(0.01, dtype=torch.float32, device=device)

        self.avogadro_const = torch.tensor(constants.value(u'Avogadro constant'), dtype=torch.float32, device=device)
        self.boltzmann_const = torch.tensor(constants.value(u'Boltzmann constant'), dtype=torch.float32, device=device)

        self.n_bond_class = len(self.ff.bond_params())
        self.n_angle_class = len(self.ff.angle_params())
        self.n_dih_class = len(self.ff.dih_params())
        self.n_lj_class = len(self.ff.lj_params())

    def convert_to_joule(self, energy):
        #converts from kJ/mol to J
        return energy * 1000.0 / self.avogadro_const

    def bond(self, atoms, indices):
        ndx1 = indices[:, :, 1] # (BS, n_bonds)
        ndx2 = indices[:, :, 2]

        param_ndx = indices[:, :, 0]

        #test = params.select(0, param_ndx)
        param = self.bond_params[param_ndx]

        #param = torch.gather(params, 0, param_ndx)
        a_0 = param[:, :, 0]
        f_c = param[:, :, 1]

        #test = atoms[ndx1, :]
        pos1 = torch.stack([a[n] for n, a in zip(ndx1, atoms)])
        pos2 = torch.stack([a[n] for n, a in zip(ndx2, atoms)])


        #pos1 = torch.gather(atoms, 1, ndx1) # (BS, n_bonds, 3)
        #pos2 = torch.gather(atoms, 1, ndx2)

        #tf.print(f_c, output_stream=sys.stdout)

        dis = pos1 - pos2
        dis = dis**2
        dis = torch.sum(dis, 2)
        dis = torch.sqrt(dis)

        #dis = tf.clip_by_value(dis, 10E-8, 1000.0)
        dis = torch.where(dis > self.bond_min_dist, dis, self.bond_min_dist)

        en = dis - a_0
        en = en**2
        en = en * f_c / 2.0
        en = torch.sum(en, 1)

        return en


    def angle(self, atoms, indices):
        ndx1 = indices[:, :, 1] # (BS, n_angles)
        ndx2 = indices[:, :, 2]
        ndx3 = indices[:, :, 3]
        param_ndx = indices[:, :, 0]

        param = self.angle_params[param_ndx]
        a_0 = param[:, :, 0]
        f_c = param[:, :, 1]

        pos1 = torch.stack([a[n] for n, a in zip(ndx1, atoms)])
        pos2 = torch.stack([a[n] for n, a in zip(ndx2, atoms)])
        pos3 = torch.stack([a[n] for n, a in zip(ndx3, atoms)])

        vec1 = pos1 - pos2
        vec2 = pos3 - pos2

        norm1 = vec1**2
        norm1 = torch.sum(norm1, dim=2)
        norm1 = torch.sqrt(norm1)
        norm2 = vec2**2
        norm2 = torch.sum(norm2, dim=2)
        norm2 = torch.sqrt(norm2)
        norm = norm1 * norm2

        dot = vec1 * vec2
        dot = torch.sum(dot, dim=2)

        #norm = tf.clip_by_value(norm, 10E-8, 1000.0)

        a = dot / norm
        a = torch.clamp(a, -0.9999, 0.9999)
        #a = tf.clip_by_value(a, -0.9999, 0.9999)  # prevent nan because of rounding errors

        a = torch.acos(a)

        en = f_c/2.0*(a - a_0)**2
        #en = en**2
        #en = en * f_c
        #en = en / 2.0
        en = torch.sum(en, dim=1)
        return en


    def dih(self, atoms, indices):
        ndx1 = indices[:, :, 1] # (BS, n_dihs)
        ndx2 = indices[:, :, 2]
        ndx3 = indices[:, :, 3]
        ndx4 = indices[:, :, 4]
        param_ndx = indices[:, :, 0]

        param = self.dih_params[param_ndx]
        a_0 = param[:, :, 0]
        f_c = param[:, :, 1]
        func_type = param[:, :, 2].type(torch.int32)
        mult = param[:, :, 3]



        pos1 = torch.stack([a[n] for n, a in zip(ndx1, atoms)])
        pos2 = torch.stack([a[n] for n, a in zip(ndx2, atoms)])
        pos3 = torch.stack([a[n] for n, a in zip(ndx3, atoms)])
        pos4 = torch.stack([a[n] for n, a in zip(ndx4, atoms)])

        vec1 = pos2 - pos1
        vec2 = pos2 - pos3
        vec3 = pos4 - pos3

        plane1 = torch.cross(vec1, vec2)
        plane2 = torch.cross(vec2, vec3)

        norm1 = plane1**2
        norm1 = torch.sum(norm1, dim=2)
        norm1 = torch.sqrt(norm1)

        norm2 = plane2**2
        norm2 = torch.sum(norm2, dim=2)
        norm2 = torch.sqrt(norm2)

        dot = plane1 * plane2
        dot = torch.sum(dot, dim=2)

        norm = norm1 * norm2 #+ 1E-20
        a = dot / norm
        a = torch.clamp(a, -0.9999, 0.9999)
        #a = torch.clamp(a, -1.0, 1.0)

        a = torch.acos(a)

        a = torch.where(func_type == 1, 3*a, a)

        en = a - a_0

        en = torch.where(func_type == 1, (torch.cos(en)+ 1.0 ) * f_c, en**2 * f_c / 2.0)

        en = torch.sum(en, dim=1)
        return en

    def lj(self, atoms, indices):
        ndx1 = indices[:, :, 1] # (BS, n_ljs)
        ndx2 = indices[:, :, 2]
        param_ndx = indices[:, :, 0]

        param = self.lj_params[param_ndx]
        sigma = param[:, :, 0]
        epsilon = param[:, :, 1]

        pos1 = torch.stack([a[n] for n, a in zip(ndx1, atoms)]) # (BS, n_ljs, 3)
        pos2 = torch.stack([a[n] for n, a in zip(ndx2, atoms)])
        dis = torch.sqrt(torch.sum((pos1 - pos2)**2, dim=2))

        dis = torch.where(dis > self.lj_min_dist, dis, self.lj_min_dist)

        c6 = torch.pow(sigma / dis, 6)
        c12 = torch.pow(c6, 2)

        en = 4 * epsilon * (c12 - c6)

        #cutoff
        #c6_cut = sigma
        #c6_cut = torch.pow(c6_cut, 6)
        #c12_cut = torch.pow(c6_cut, 2)
        #en_cut = 4 * epsilon * (c12_cut - c6_cut)
        #en = en - en_cut
        #en = torch.where(dis <= 1.0, en, torch.tensor(0.0))

        en = torch.sum(en, dim=1)

        return en

    def bond_grid(self, pos_grid, atoms, indices):
        #ndx1 = indices[:, :, 1] # (BS, n_ljs)

        ndx2 = indices[:, :, 2]
        param_ndx = indices[:, :, 0]

        print(param_ndx)

        param = self.bond_params[param_ndx]
        a_0 = param[:, :, 0, None, None, None]
        f_c = param[:, :, 1, None, None, None]

        pos2 = torch.stack([a[n] for n, a in zip(ndx2, atoms)])[:,:,:, None, None, None]

        dis = torch.sqrt(torch.sum((pos_grid - pos2)**2, dim=2))

        dis = torch.where(dis > self.bond_min_dist, dis, self.bond_min_dist)

        en = dis - a_0
        en = en**2

        en = en * f_c / 2.0

        en = torch.sum(en, dim=1)
        en = torch.exp(-en * 1000.0 / (self.avogadro_const * self.boltzmann_const*568))
        #en_sum = torch.sum(en, dim=[1, 2, 3])[:, None, None, None] + 1E-12
        en_sum, _ = torch.max(en.view(en.size(0), -1), -1)
        en_sum = en_sum[:, None, None, None] + 1E-12
        #print(en_sum)
        en = en / en_sum

        return en


    def angle_grid(self, pos_grid, atoms, indices):
        ndx2 = indices[:, :, 2]
        ndx3 = indices[:, :, 3]

        param_ndx = indices[:, :, 0]

        param = self.angle_params[param_ndx]
        a_0 = param[:, :, 0, None, None, None]
        f_c = param[:, :, 1, None, None, None]

        pos2 = torch.stack([a[n] for n, a in zip(ndx2, atoms)])[:,:,:, None, None, None]
        pos3 = torch.stack([a[n] for n, a in zip(ndx3, atoms)])[:,:,:, None, None, None]

        #print(pos2)
        vec1 = pos_grid - pos2
        vec2 = pos3 - pos2

        norm1 = vec1**2
        norm1 = torch.sum(norm1, dim=2)
        norm1 = torch.sqrt(norm1)
        norm2 = vec2**2
        norm2 = torch.sum(norm2, dim=2)
        norm2 = torch.sqrt(norm2)
        norm = norm1 * norm2

        dot = vec1 * vec2
        dot = torch.sum(dot, dim=2)

        a = dot / norm
        a = torch.clamp(a, -0.9999, 0.9999)

        a = torch.acos(a)

        en = f_c/2.0*(a - a_0)**2

        en = torch.sum(en, dim=1)

        en = torch.exp(-en * 1000.0 / (self.avogadro_const * self.boltzmann_const*568))
        #en_sum = torch.sum(en, dim=[1, 2, 3])[:, None, None, None] + 1E-12
        en_sum, _ = torch.max(en.view(en.size(0), -1), -1)
        en_sum = en_sum[:, None, None, None] + 1E-12
        #print(en_sum)
        en = en / en_sum

        return en



    def dih_grid(self, pos_grid, atoms, indices):
        ndx2 = indices[:, :, 2]
        ndx3 = indices[:, :, 3]
        ndx4 = indices[:, :, 4]

        param_ndx = indices[:, :, 0]

        param = self.dih_params[param_ndx]
        a_0 = param[:, :, 0, None, None, None]
        f_c = param[:, :, 1, None, None, None]
        func_type = param[:, :, 2].type(torch.int32)[:,:, None, None, None]
        mult = param[:, :, 3]

        pos2 = torch.stack([a[n] for n, a in zip(ndx2, atoms)])[:,:,:, None, None, None]
        pos3 = torch.stack([a[n] for n, a in zip(ndx3, atoms)])[:,:,:, None, None, None]
        pos4 = torch.stack([a[n] for n, a in zip(ndx4, atoms)])[:,:,:, None, None, None]

        vec1 = pos2 - pos_grid
        vec2 = pos2 - pos3
        vec3 = pos4 - pos3

        plane1 = torch.cross(vec1, vec2.repeat(1,1,1,16,16,16))
        plane2 = torch.cross(vec2, vec3)

        norm1 = plane1**2
        norm1 = torch.sum(norm1, dim=2)
        norm1 = torch.sqrt(norm1)


        norm2 = plane2**2
        norm2 = torch.sum(norm2, dim=2)
        norm2 = torch.sqrt(norm2)

        dot = plane1 * plane2
        dot = torch.sum(dot, dim=2)

        norm = norm1 * norm2 #+ 1E-20

        a = dot / norm
        a = torch.clamp(a, -0.9999, 0.9999)

        a = torch.acos(a)

        a = torch.where(func_type == 1, 3*a, a)

        en = a - a_0

        en = torch.where(func_type == 1, (torch.cos(en)+ 1.0 ) * f_c, en**2 * f_c / 2.0)

        en = torch.sum(en, dim=1)

        en = torch.exp(-en * 1000.0 / (self.avogadro_const * self.boltzmann_const*568))
        #en_sum = torch.sum(en, dim=[1, 2, 3])[:, None, None, None] + 1E-12
        en_sum, _ = torch.max(en.view(en.size(0), -1), -1)
        en_sum = en_sum[:, None, None, None] + 1E-12
        #print(en_sum)
        #print(en)
        en = en / en_sum
        #print(en)

        return en


    def lj_grid(self, pos_grid, atoms, indices):
        #ndx1 = indices[:, :, 1] # (BS, n_ljs)

        ndx2 = indices[:, :, 2]
        param_ndx = indices[:, :, 0]

        param = self.lj_params[param_ndx]
        sigma = param[:, :, 0, None, None, None]
        epsilon = param[:, :, 1, None, None, None]

        # pos_grid (1, 1, 3, N_x, N_y, N_z)

        #pos1 = torch.stack([a[n] for n, a in zip(ndx1, atoms)]) # (BS, n_ljs, 3)
        pos2 = torch.stack([a[n] for n, a in zip(ndx2, atoms)])[:,:,:, None, None, None]


        dis = torch.sqrt(torch.sum((pos_grid - pos2)**2, dim=2))

        dis = torch.where(dis > self.lj_min_dist, dis, self.lj_min_dist)

        c6 = torch.pow(sigma / dis, 6)
        c12 = torch.pow(c6, 2)

        en = 4 * epsilon * (c12 - c6)

        #cutoff
        #c6_cut = sigma
        #c6_cut = torch.pow(c6_cut, 6)
        #c12_cut = torch.pow(c6_cut, 2)
        #en_cut = 4 * epsilon * (c12_cut - c6_cut)
        #en = en - en_cut
        #en = torch.where(dis <= 1.0, en, torch.tensor(0.0))

        en = torch.sum(en, dim=1)

        en = torch.exp(-en * 1000.0 / (self.avogadro_const * self.boltzmann_const*568))
        #en_sum = torch.sum(en, dim=[1, 2, 3])[:, None, None, None] + 1E-12
        en_sum, _ = torch.max(en.view(en.size(0), -1), -1)
        en_sum = en_sum[:, None, None, None] + 1E-12
        #print(en_sum)
        en = en / en_sum

        return en


    def bond_dstr(self, atoms, indices, n_bins=20, bin_width=0.01, bin_start=0.0):
        gauss_centers = torch.tensor(np.arange(bin_start, bin_start+n_bins*bin_width, bin_width), dtype=torch.float32, device=self.device)[None, None, :]
        ndx1 = indices[:, :, 1] # (BS, n_bonds)
        ndx2 = indices[:, :, 2]
        param_ndx = indices[:, :, 0]
        #param_ndx = torch.where(param_ndx >= 0, param_ndx, torch.max(param_ndx)+1)
        #param_ndx = param_ndx - torch.min(param_ndx)
        param_ndx = param_ndx + 1
        param_one_hot = torch.nn.functional.one_hot(param_ndx, num_classes=self.n_bond_class)[:,:,1:] #(BS, N_bonds, N_classes)

        pos1 = torch.stack([a[n] for n, a in zip(ndx1, atoms)])
        pos2 = torch.stack([a[n] for n, a in zip(ndx2, atoms)])
        dis = pos1 - pos2
        dis = dis**2
        dis = torch.sum(dis, 2)
        dis = torch.sqrt(dis)[:,:, None] #(BS, N_bonds, 1)
        histo = (gauss_centers - dis)**2 #(BS, N_bonds, N_bins)
        histo = torch.exp(-histo/(2*bin_width**2))
        histo = histo[:,:,:,None] * param_one_hot[:,:,None,:] #(BS, N_bonds, N_bins, N_classes)
        histo = torch.sum(histo, 1) #(BS, N_bins, N_classes)
        histo = histo / (torch.sum(histo, 1)[:, None, :] + 1E-20)

        return histo

    def lj_dstr(self, atoms, indices, n_bins=20, bin_width=0.025, bin_start=0.0):
        gauss_centers = torch.tensor(np.arange(bin_start, bin_start+n_bins*bin_width, bin_width), dtype=torch.float32, device=self.device)[None, None, :]
        ndx1 = indices[:, :, 1] # (BS, n_bonds)
        ndx2 = indices[:, :, 2]
        param_ndx = indices[:, :, 0]
        #param_ndx = torch.where(param_ndx >= 0, param_ndx, torch.max(param_ndx)+1)
        #param_ndx = param_ndx - torch.min(param_ndx)
        param_ndx = param_ndx + 1
        param_one_hot = torch.nn.functional.one_hot(param_ndx, num_classes=self.n_lj_class)[:,:,1:] #(BS, N_bonds, N_classes)

        pos1 = torch.stack([a[n] for n, a in zip(ndx1, atoms)])
        pos2 = torch.stack([a[n] for n, a in zip(ndx2, atoms)])
        dis = pos1 - pos2
        dis = dis**2
        dis = torch.sum(dis, 2)
        dis = torch.sqrt(dis)[:,:, None] #(BS, N_bonds, 1)
        histo = (gauss_centers - dis)**2 #(BS, N_bonds, N_bins)
        histo = torch.exp(-histo/(2*bin_width**2))
        histo = histo[:,:,:,None] * param_one_hot[:,:,None,:] #(BS, N_bonds, N_bins, N_classes)
        histo = torch.sum(histo, 1) #(BS, N_bins, N_classes)
        histo = histo / (torch.sum(histo, 1)[:, None, :] + 1E-20)
        return histo

    def angle_dstr(self, atoms, indices, n_bins=20, bin_width=9.0, bin_start=0.0):
        gauss_centers = torch.tensor(np.arange(bin_start, bin_start+n_bins*bin_width, bin_width), dtype=torch.float32, device=self.device)[None, None, :]
        ndx1 = indices[:, :, 1] # (BS, n_angles)
        ndx2 = indices[:, :, 2]
        ndx3 = indices[:, :, 3]
        pos1 = torch.stack([a[n] for n, a in zip(ndx1, atoms)])
        pos2 = torch.stack([a[n] for n, a in zip(ndx2, atoms)])
        pos3 = torch.stack([a[n] for n, a in zip(ndx3, atoms)])
        vec1 = pos1 - pos2
        vec2 = pos3 - pos2
        norm1 = vec1**2
        norm1 = torch.sum(norm1, dim=2)
        norm1 = torch.sqrt(norm1)
        norm2 = vec2**2
        norm2 = torch.sum(norm2, dim=2)
        norm2 = torch.sqrt(norm2)
        norm = norm1 * norm2
        dot = vec1 * vec2
        dot = torch.sum(dot, dim=2)
        a = dot / norm
        a = torch.clamp(a, -0.9999, 0.9999)
        a = torch.acos(a)[:,:,None]
        a = a * 180. / np.pi
        param_ndx = indices[:, :, 0]
        param_ndx = param_ndx + 1
        param_one_hot = torch.nn.functional.one_hot(param_ndx, num_classes=self.n_angle_class)[:,:,1:] #(BS, N_bonds, N_classes)

        histo = (gauss_centers - a)**2 #(BS, N_bonds, N_bins)
        histo = torch.exp(-histo/(2*bin_width**2))
        histo = histo[:,:,:,None] * param_one_hot[:,:,None,:] #(BS, N_bonds, N_bins, N_classes)
        histo = torch.sum(histo, 1) #(BS, N_bins, N_classes)
        histo = histo / (torch.sum(histo, 1)[:, None, :] + 1E-20)

        return histo

    def dih_dstr(self, atoms, indices, n_bins=20, bin_width=9.0, bin_start=0.0):
        gauss_centers = torch.tensor(np.arange(bin_start, bin_start+n_bins*bin_width, bin_width), dtype=torch.float32, device=self.device)[None, None, :]
        ndx1 = indices[:, :, 1] # (BS, n_dihs)
        ndx2 = indices[:, :, 2]
        ndx3 = indices[:, :, 3]
        ndx4 = indices[:, :, 4]
        pos1 = torch.stack([a[n] for n, a in zip(ndx1, atoms)])
        pos2 = torch.stack([a[n] for n, a in zip(ndx2, atoms)])
        pos3 = torch.stack([a[n] for n, a in zip(ndx3, atoms)])
        pos4 = torch.stack([a[n] for n, a in zip(ndx4, atoms)])
        vec1 = pos2 - pos1
        vec2 = pos2 - pos3
        vec3 = pos4 - pos3
        plane1 = torch.cross(vec1, vec2)
        plane2 = torch.cross(vec2, vec3)
        norm1 = plane1**2
        norm1 = torch.sum(norm1, dim=2)
        norm1 = torch.sqrt(norm1)
        norm2 = plane2**2
        norm2 = torch.sum(norm2, dim=2)
        norm2 = torch.sqrt(norm2)
        dot = plane1 * plane2
        dot = torch.sum(dot, dim=2)
        norm = norm1 * norm2 #+ 1E-20
        a = dot / norm
        a = torch.clamp(a, -0.9999, 0.9999)
        a = torch.acos(a)[:,:,None]
        a = a * 180. / np.pi


        param_ndx = indices[:, :, 0]
        param_ndx = param_ndx + 1
        param_one_hot = torch.nn.functional.one_hot(param_ndx, num_classes=self.n_dih_class)[:,:,1:] #(BS, N_bonds, N_classes)

        histo = (gauss_centers - a)**2 #(BS, N_bonds, N_bins)
        histo = torch.exp(-histo/(2*bin_width**2))
        histo = histo[:,:,:,None] * param_one_hot[:,:,None,:] #(BS, N_bonds, N_bins, N_classes)
        histo = torch.sum(histo, 1) #(BS, N_bins, N_classes)
        histo = histo / (torch.sum(histo, 1)[:, None, :] + 1E-20)

        return histo
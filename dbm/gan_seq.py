import torch
from torch.optim import Adam, RMSprop
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from dbm.util import make_grid_np, rand_rotation_matrix, voxelize_gauss, make_dir, avg_blob, voxelize_gauss_batch
from dbm.torch_energy import *
from dbm.output import *
from dbm.recurrent_generator import Generator
from tqdm import tqdm
import numpy as np
#from tqdm import tqdm
from timeit import default_timer as timer
import os
import math
#from configparser import ConfigParser
#import mdtraj as md
#from universe import *
import dbm.model as model
from dbm.data import *
from scipy import constants
#import dbm.tf_utils as tf_utils
#import dbm.tf_energy as tf_energy
from copy import deepcopy
from shutil import copyfile
from contextlib import redirect_stdout
from operator import add
from itertools import cycle

#tf.compat.v1.disable_eager_execution()

torch.set_default_dtype(torch.float32)


def rand_rot_mat(align):
    # rotation axis
    if align:
        v_rot = np.array([0.0, 0.0, 1.0])
    else:
        phi = np.random.uniform(0, np.pi * 2)
        costheta = np.random.uniform(-1, 1)
        theta = np.arccos(costheta)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        v_rot = np.array([x, y, z])

    # rotation angle
    theta = np.random.uniform(0, np.pi * 2)

    # rotation matrix
    a = math.cos(theta / 2.0)
    b, c, d = -v_rot * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    rot_mat = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

    return rot_mat


class DS(Dataset):
    def __init__(self, data, cfg):

        generators = []
        generators.append(Generator(data, hydrogens=False, gibbs=False, train=True, rand_rot=False))
        generators.append(Generator(data, hydrogens=True, gibbs=False, train=True, rand_rot=False))
        generators.append(Generator(data, hydrogens=False, gibbs=True, train=True, rand_rot=False))
        generators.append(Generator(data, hydrogens=True, gibbs=True, train=True, rand_rot=False))

        self.elems = []
        for g in generators:
            self.elems += g.all_elems()

        self.resolution = cfg.getint('grid', 'resolution')
        self.delta_s = cfg.getfloat('grid', 'length') / cfg.getint('grid', 'resolution')
        self.sigma = cfg.getfloat('grid', 'sigma')

        if cfg.getboolean('training', 'rand_rot'):
            self.rand_rot = True
            print("using random rotations during training...")
        else:
            self.rand_rot = False

        self.grid = make_grid_np(self.delta_s, self.resolution)


    def __len__(self):
        return len(self.elems)

    def __getitem__(self, ndx):
        if self.rand_rot:
            R = rand_rot_mat(self.data.align)
        else:
            R = np.eye(3)

        _, d = self.elems[ndx]



        #item = self.array(self.elems[ndx][1:], np.float32)
        #target_pos, target_type, aa_feat, repl, mask, aa_pos, cg_pos, cg_feat, *energy_ndx = item
        #energy_ndx = self.array(energy_ndx, np.int64)

        target_atom = voxelize_gauss(np.dot(d['target_pos'], R.T), self.sigma, self.grid)
        atom_grid = voxelize_gauss(np.dot(d['aa_pos'], R.T), self.sigma, self.grid)
        bead_grid = voxelize_gauss(np.dot(d['cg_pos'], R.T), self.sigma, self.grid)

        cg_features = d['cg_feat'][:, :, None, None, None] * bead_grid[:, None, :, :, :]
        # (N_beads, N_chn, 1, 1, 1) * (N_beads, 1, N_x, N_y, N_z)
        cg_features = np.sum(cg_features, 0)

        elems = (target_atom, d['target_type'], d['aa_feat'], d['repl'], d['mask'])
        initial = (atom_grid, cg_features)
        energy_ndx = (d['bonds_ndx'], d['angles_ndx'], d['dihs_ndx'], d['ljs_ndx'])

        #energy_ndx = (bonds_ndx, angles_ndx, dihs_ndx, ljs_ndx)

        #return atom_grid, bead_grid, target_atom, target_type, aa_feat, repl, mask, energy_ndx
        #return atom_grid, cg_features, target_atom, d['target_type'], d['aa_feat'], d['repl'], d['mask'], energy_ndx, d['aa_pos']
        return elems, initial, energy_ndx


    def array(self, elems, dtype):
        return tuple(np.array(t, dtype=dtype) for t in elems)

class GAN_SEQ():

    def __init__(self, device, cfg):

        self.device = device
        self.cfg = cfg

        self.bs = self.cfg.getint('training', 'batchsize')

        #Data pipeline
        self.data = Data(cfg, save=False)
        ds_train = DS(self.data, cfg)
        self.loader_train = DataLoader(
            ds_train,
            batch_size=self.bs,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            num_workers=0,
        )

        self.data = ds_train.data
        self.ff = ds_train.data.ff

        ds_val = DS(self.cfg, train=False)
        loader_val = DataLoader(
            ds_val,
            batch_size=self.bs,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            num_workers=0,
        )
        self.loader_val = cycle(loader_val)
        self.val_data = ds_val.data

        self.n_gibbs = int(cfg.getint('validate', 'n_gibbs'))

        #model
        self.name = cfg.get('model', 'name')
        self.z_dim = int(cfg.getint('model', 'noise_dim'))
        self.n_atom_chns = self.ff.n_atom_chns
        self.z_and_label_dim = self.z_dim + self.n_atom_chns

        self.step = 0
        self.epoch = 1

        # Make Dirs for saving
        self.out = OutputHandler(
            self.name,
            self.cfg.getint('training', 'n_checkpoints'),
            self.cfg.get('model', 'output_dir'),
        )
        #forcefield for energy loss

        self.energy = Energy(self.ff, self.device)

        #self.bond_params = torch.tensor(self.ff.bond_params(), dtype=torch.float32, device=device)
        #self.angle_params = torch.tensor(self.ff.angle_params(), dtype=torch.float32, device=device)
        #self.dih_params = torch.tensor(self.ff.dih_params(), dtype=torch.float32, device=device)
        #self.lj_params = torch.tensor(self.ff.lj_params(), dtype=torch.float32, device=device)

        #self.atom_mass = torch.tensor([[[atype.mass for atype in self.ff.atom_types.values()]]], dtype=torch.float32, device=device) #(1, 1, n_atomtypes)


        #print(self.energy_prior_weights)
        #print(self.energy_prior_steps)
        self.prior_weights = self.get_prior_weights()
        self.lj_weight = cfg.getfloat('training', 'lj_weight')
        self.covalent_weight = cfg.getfloat('training', 'covalent_weight')
        self.energy_prior_mode = int(cfg.getint('training', 'energy_prior_mode'))

        #print(self.prior_weights)

        self.w_prior = torch.tensor(self.prior_weights[self.step], dtype=torch.float32, device=device)

        #Model selection
        if cfg.get('model', 'model_type') == "small":
            print("Using small model")
            self.critic = model.AtomCrit_small(in_channels=self.ff.n_channels+1, fac=1, sn=self.cfg.getint('model', 'sn_crit'), device=device)
            self.generator = model.AtomGen_small(self.z_and_label_dim, condition_n_channels=self.ff.n_channels, fac=1, sn=self.cfg.getint('model', 'sn_gen'), device=device)
        else:
            print("Using big model")
            self.critic = model.AtomCrit2(in_channels=self.ff.n_channels + 1, fac=1,
                                          sn=self.cfg.getint('model', 'sn_crit'), device=device)
            self.generator = model.AtomGen2(self.z_and_label_dim, condition_n_channels=self.ff.n_channels, fac=1,
                                            sn=self.cfg.getint('model', 'sn_gen'), device=device)

        self.use_gp = cfg.getboolean('model', 'gp')

        self.mse = torch.nn.MSELoss()
        self.kld = torch.nn.KLDivLoss(reduction="batchmean")

        self.critic.to(device=device)
        self.generator.to(device=device)
        self.mse.to(device=device)

        self.opt_generator_pretrain = Adam(self.generator.parameters(), lr=0.00005, betas=(0, 0.9))
        self.opt_generator = Adam(self.generator.parameters(), lr=0.00005, betas=(0, 0.9))
        self.opt_critic = Adam(self.critic.parameters(), lr=0.0001, betas=(0, 0.9))


        self.restored_model = False
        self.restore_latest_checkpoint()

    def get_prior_weights(self):
        steps_per_epoch = len(self.loader_train)
        tot_steps = steps_per_epoch * self.cfg.getint('training', 'n_epoch')

        prior_weights = self.cfg.get('training', 'energy_prior_weights')
        prior_weights = [float(v) for v in prior_weights.split(",")]
        prior_steps = self.cfg.get('training', 'n_start_prior')
        prior_steps = [int(v) for v in prior_steps.split(",")]
        n_trans = self.cfg.getint('training', 'n_prior_transition')
        weights = []
        for s in range(self.step, self.step + tot_steps):
            if s > prior_steps[-1]:
                ndx = len(prior_weights)-1
                #weights.append(self.energy_prior_values[-1])
            else:
                for n in range(0, len(prior_steps)):
                    if s < prior_steps[n]:
                        #weights.append(self.energy_prior_values[n])
                        ndx = n
                        break
            #print(ndx)
            if ndx > 0 and s < prior_steps[ndx-1] + self.cfg.getint('training', 'n_prior_transition'):
                weights.append(prior_weights[ndx-1] + (prior_weights[ndx]-prior_weights[ndx-1])*(s-prior_steps[ndx-1])/n_trans)
            else:
                weights.append(prior_weights[ndx])

        return weights

    def make_checkpoint(self):
        return self.out.make_checkpoint(
            self.step,
            {
                "generator": self.generator.state_dict(),
                "critic": self.critic.state_dict(),
                "opt_generator": self.opt_generator.state_dict(),
                "opt_critic": self.opt_critic.state_dict(),
                "step": self.step,
                "epoch": self.epoch,
            },
        )

    def restore_latest_checkpoint(self):
        latest_ckpt = self.out.latest_checkpoint()
        if latest_ckpt is not None:
            checkpoint = torch.load(latest_ckpt)
            self.generator.load_state_dict(checkpoint["generator"])
            self.critic.load_state_dict(checkpoint["critic"])
            self.opt_generator.load_state_dict(checkpoint["opt_generator"])
            self.opt_critic.load_state_dict(checkpoint["opt_critic"])
            self.step = checkpoint["step"]
            self.epoch = checkpoint["epoch"]
            self.restored_model = True
            print("restored model!!!")

        self.out.prune_checkpoints()

    #def map_to_device(self, tup):
    #    return tuple(x.to(device=self.device) for x in tup)

    def map_to_device(self, tup):
        return tuple(tuple(y.to(device=self.device) for y in x) if type(x) is list else x.to(device=self.device) for x in tup)

    def transpose_and_zip(self, *args):
        args = tuple(torch.transpose(x, 0, 1) for x in args)
        elems = zip(*args)
        return elems

    def featurize(self, grid, features):
        grid = grid[:, :, None, :, :, :] * features[:, :, :, None, None, None]
        #grid (BS, N_atoms, 1, N_x, N_y, N_z) * features (BS, N_atoms, N_features, 1, 1, 1)
        return torch.sum(grid, 1)

    def prepare_condition(self, fake_atom_grid, real_atom_grid, aa_featvec, bead_features):
        fake_aa_features = self.featurize(fake_atom_grid, aa_featvec)
        real_aa_features = self.featurize(real_atom_grid, aa_featvec)
        c_fake = fake_aa_features + bead_features
        c_real = real_aa_features + bead_features
        return c_fake, c_real

    def generator_loss(self, critic_fake):
        return (-1.0 * critic_fake).mean()

    def critic_loss(self, critic_real, critic_fake):
        loss_on_generated = critic_fake.mean()
        loss_on_real = critic_real.mean()

        loss = loss_on_generated - loss_on_real
        return loss

    def epsilon_penalty(self, epsilon, critic_real_outputs):
        if epsilon > 0:
            penalties = torch.pow(critic_real_outputs, 2)
            penalty = epsilon * penalties.mean()
            return penalty
        return 0.0

    def gradient_penalty(self, real_data, fake_data, mask):
        batch_size = real_data.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1, 1, 1, device=self.device)
        alpha = alpha.expand_as(real_data)

        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated = Variable(interpolated, requires_grad=True)
        interpolated.to(self.device)

        # Calculate probability of interpolated examples
        prob_interpolated = self.critic(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size(), device=self.device),
                               create_graph=True, retain_graph=True)[0]

        gradients = gradients.view(batch_size, -1)

        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        gradients_norm = ((gradients_norm - 1) ** 2)
        gradients_norm = gradients_norm * mask

        # Return gradient penalty
        return gradients_norm.mean()

    def get_energies_from_grid(self, atom_grid, energy_ndx):
        coords = avg_blob(
            atom_grid,
            res=self.cfg.getint('grid', 'resolution'),
            width=self.cfg.getfloat('grid', 'length'),
            sigma=self.cfg.getfloat('grid', 'sigma'),
            device=self.device,
        )
        bond_ndx, angle_ndx, dih_ndx, lj_ndx = energy_ndx
        b_energy = self.energy.bond(coords, bond_ndx)
        a_energy = self.energy.angle(coords, angle_ndx)
        d_energy = self.energy.dih(coords, dih_ndx)
        l_energy = self.energy.lj(coords, lj_ndx)

        return b_energy, a_energy, d_energy, l_energy

    def get_energies_from_coords(self, coords, energy_ndx):

        bond_ndx, angle_ndx, dih_ndx, lj_ndx = energy_ndx
        b_energy = self.energy.bond(coords, bond_ndx)
        a_energy = self.energy.angle(coords, angle_ndx)
        d_energy = self.energy.dih(coords, dih_ndx)
        l_energy = self.energy.lj(coords, lj_ndx)

        return b_energy, a_energy, d_energy, l_energy

    def get_forces(self, x, energy_ndx):
        x = x.requires_grad_(True)
        b_energy, angle_energy, dih_energy, lj_energy = self.get_energies_from_coords(x, energy_ndx)
        energy = b_energy + angle_energy + dih_energy + lj_energy
        #for f in torch.autograd.grad(energy, x, torch.ones_like(energy), create_graph=True, retain_graph=True):
        #    print(f.size())
        return -torch.autograd.grad(energy, x, torch.ones_like(energy), create_graph=True, retain_graph=True)[0]

    def force_matching_loss(self, real_coords, fake_atom_grid, energy_ndx):
        fake_coords = avg_blob(
            fake_atom_grid,
            res=self.cfg.getint('grid', 'resolution'),
            width=self.cfg.getfloat('grid', 'length'),
            sigma=self.cfg.getfloat('grid', 'sigma'),
            device=self.device,
        )

        fake_forces = self.get_forces(fake_coords, energy_ndx)
        real_forces = self.get_forces(real_coords, energy_ndx)

        force_loss = torch.mean((fake_forces - real_forces)**2)

        fb, fa, fd, fl = self.get_energies_from_coords(fake_coords, energy_ndx)

        return force_loss, torch.mean(fb), torch.mean(fa), torch.mean(fd), torch.mean(fl)

    def energy_min_loss(self, atom_grid, energy_ndx):

        fb, fa, fd, fl = self.get_energies_from_grid(atom_grid, energy_ndx)

        b_loss = torch.mean(fb)
        a_loss = torch.mean(fa)
        d_loss = torch.mean(fd)
        l_loss = torch.mean(fl)

        return b_loss, a_loss, d_loss, l_loss

    def truncated_energy_min_loss(self, real_atom_grid, fake_atom_grid, energy_ndx):

        rb, ra, rd, rl = self.get_energies_from_grid(real_atom_grid, energy_ndx)
        fb, fa, fd, fl = self.get_energies_from_grid(fake_atom_grid, energy_ndx)

        b_dif = fb - rb
        a_dif = fa - ra
        d_dif = fd - rd
        l_dif = fl - rl

        b_loss = torch.mean(torch.max(b_dif, torch.tensor(0, dtype=torch.float32, device=self.device)))
        a_loss = torch.mean(torch.max(a_dif, torch.tensor(0, dtype=torch.float32, device=self.device)))
        d_loss = torch.mean(torch.max(d_dif, torch.tensor(0, dtype=torch.float32, device=self.device)))
        l_loss = torch.mean(torch.max(l_dif, torch.tensor(0, dtype=torch.float32, device=self.device)))

        return b_loss, a_loss, d_loss, l_loss, torch.mean(fb), torch.mean(fa), torch.mean(fd), torch.mean(fl)

    def energy_loss_mean_abs_dif(self, real_atom_grid, fake_atom_grid, energy_ndx):

        rb, ra, rd, rl = self.get_energies_from_grid(real_atom_grid, energy_ndx)
        fb, fa, fd, fl = self.get_energies_from_grid(fake_atom_grid, energy_ndx)

        b_loss = torch.mean(torch.abs(rb - fb))
        a_loss = torch.mean(torch.abs(ra - fa))
        d_loss = torch.mean(torch.abs(rd - fd))
        l_loss = torch.mean(torch.abs(rl - fl))

        return b_loss, a_loss, d_loss, l_loss, torch.mean(fb), torch.mean(fa), torch.mean(fd), torch.mean(fl)



    def energy_loss_mean_sq_dif(self, real_atom_grid, fake_atom_grid, energy_ndx):

        rb, ra, rd, rl = self.get_energies_from_grid(real_atom_grid, energy_ndx)
        fb, fa, fd, fl = self.get_energies_from_grid(fake_atom_grid, energy_ndx)

        b_loss = torch.mean((rb - fb)**2)
        a_loss = torch.mean((ra - fa)**2)
        d_loss = torch.mean((rd - fd)**2)
        l_loss = torch.mean((rl - fl)**2)

        return b_loss, a_loss, d_loss, l_loss, torch.mean(fb), torch.mean(fa), torch.mean(fd), torch.mean(fl)

    def energy_loss_mean_abs_dif2(self, real_coords, fake_atom_grid, energy_ndx):

        rb, ra, rd, rl = self.get_energies_from_coords(real_coords, energy_ndx)
        fb, fa, fd, fl = self.get_energies_from_grid(fake_atom_grid, energy_ndx)

        b_loss = torch.mean(torch.abs(rb - fb))
        a_loss = torch.mean(torch.abs(ra - fa))
        d_loss = torch.mean(torch.abs(rd - fd))
        l_loss = torch.mean(torch.abs(rl - fl))

        return b_loss, a_loss, d_loss, l_loss, torch.mean(fb), torch.mean(fa), torch.mean(fd), torch.mean(fl)

    def free_energy_perturbation_loss(self, real_atom_grid, fake_atom_grid, energy_ndx, temp):

        rb, ra, rd, rl = self.get_energies_from_grid(real_atom_grid, energy_ndx)
        fb, fa, fd, fl = self.get_energies_from_grid(fake_atom_grid, energy_ndx)

        real_energy = rb + ra + rd + rl
        fake_energy = fb + fa + fd + fl
        energy_dif = fake_energy - real_energy
        energy_dif = self.energy.convert_to_joule(energy_dif)

        boltzmann_weights = torch.exp(- energy_dif / (self.energy.boltzmann_const * temp))
        print(energy_dif/ (self.energy.boltzmann_const * temp))
        average_boltzmann_weight = torch.mean(boltzmann_weights)

        #loss = - constants.value(u'Boltzmann constant') * temp * torch.log(average_boltzmann_weight + 1E-24)
        loss = torch.abs(torch.log(average_boltzmann_weight + 1E-40))
        #print(fake_energy, real_energy, loss)

        return loss, torch.mean(fb), torch.mean(fa), torch.mean(fd), torch.mean(fl)

    def detach(self, t):
        t = tuple([c.detach().cpu().numpy() for c in t])
        return t

    def train(self):
        steps_per_epoch = len(self.loader_train)
        n_critic = self.cfg.getint('training', 'n_critic')
        n = 0
        n_save = int(self.cfg.getint('record', 'n_save'))
        
        epochs = tqdm(range(self.epoch, self.cfg.getint('training', 'n_epoch')))
        epochs.set_description('epoch: ')
        for epoch in epochs:
            value_list = [[], [], [], [], [], [], []]
            data = tqdm(self.loader_train, total=steps_per_epoch, leave=False)
            for batch in data:
                batch = self.map_to_device(batch)
                elems, initial, energy_ndx = batch
                #(atom_grid, bead_features, target_atom, target_type, aa_feat, repl, mask, energy_ndx, aa_pos) = batch
                elems = self.transpose_and_zip(elems)

                if n == n_critic:
                    w, e, b, a, d, l = self.train_step_gen(atom_grid, bead_features, elems, energy_ndx, aa_pos)

                    c, w, e, b, a, d, l = self.detach((c, w, e, b, a, d, l))
                    for value, list in zip((c, w, e, b, a, d, l), value_list):
                        list.append(value)
                    data.set_description('D: {}, G: {}, {}, {}, {}, {}, {}'.format(c, w, e, b, a, d, l))
                    self.out.add_scalar("Generator/loss_w", w, global_step=self.step)
                    self.out.add_scalar("Generator/loss_e", e, global_step=self.step)
                    self.out.add_scalar("Generator/bond_energy", b, global_step=self.step)
                    self.out.add_scalar("Generator/angle_energy", a, global_step=self.step)
                    self.out.add_scalar("Generator/dih_energy", d, global_step=self.step)
                    self.out.add_scalar("Generator/lj_energy", l, global_step=self.step)
                    self.out.add_scalar("Critic/loss", c, global_step=self.step)

                    val_batch = next(self.loader_val)
                    val_batch = self.map_to_device(val_batch)
                    (atom_grid, bead_features, target_atom, target_type, aa_feat, repl, mask, energy_ndx, aa_pos) = val_batch
                    elems = self.transpose_and_zip(target_atom, target_type, aa_feat, repl, mask)
                    w, e, b, a, d, l = self.val_step_gen(atom_grid, bead_features, elems, energy_ndx, aa_pos)
                    self.out.add_scalar("Generator/loss_w", w, global_step=self.step, mode='val')
                    self.out.add_scalar("Generator/loss_e", e, global_step=self.step, mode='val')
                    self.out.add_scalar("Generator/bond_energy", b, global_step=self.step, mode='val')
                    self.out.add_scalar("Generator/angle_energy", a, global_step=self.step, mode='val')
                    self.out.add_scalar("Generator/dih_energy", d, global_step=self.step, mode='val')
                    self.out.add_scalar("Generator/lj_energy", l, global_step=self.step, mode='val')

                    self.step += 1
                    n = 0

                else:
                    c = self.train_step_critic(atom_grid, bead_features, elems)
                    n += 1

                #if self.step % self.cfg.getint('training', 'n_save') == 0:
                #    self.make_checkpoint()
                #    self.out.prune_checkpoints()
            tqdm.write('epoch {} steps {} : D: {} G: {}, {}, {}, {}, {}, {}'.format(
                self.epoch,
                self.step,
                sum(value_list[0])/len(value_list[0]),
                sum(value_list[1]) / len(value_list[1]),
                sum(value_list[2]) / len(value_list[2]),
                sum(value_list[3]) / len(value_list[3]),
                sum(value_list[4]) / len(value_list[4]),
                sum(value_list[5]) / len(value_list[5]),
                sum(value_list[6]) / len(value_list[6]),
            ))



            if epoch % n_save == 0:
                self.make_checkpoint()
                self.out.prune_checkpoints()
                self.validate()

            self.epoch += 1




    def train_step_critic(self, initial_atom_grid, bead_features, elems):
        c_loss = torch.zeros([], dtype=torch.float32, device=self.device)

        fake_atom_grid = initial_atom_grid.clone()
        real_atom_grid = initial_atom_grid.clone()


        for target_atom, target_type, aa_featvec, repl, mask in elems:
            #prepare input for generator
            c_fake, c_real = self.prepare_condition(fake_atom_grid, real_atom_grid, aa_featvec, bead_features)
            z = torch.empty(
                [target_atom.shape[0], self.z_dim],
                dtype=torch.float32,
                device=self.device,
            ).normal_()

            #generate fake atom
            fake_atom = self.generator(z, target_type, c_fake)

            fake_data = torch.cat([fake_atom, c_fake], dim=1)
            real_data = torch.cat([target_atom[:, None, :, :, :], c_real], dim=1)

            #critic
            critic_fake = self.critic(fake_data)
            critic_real = self.critic(real_data)

            #mask
            critic_fake = torch.squeeze(critic_fake) * mask
            critic_real = torch.squeeze(critic_real) * mask

            #loss
            c_wass = self.critic_loss(critic_real, critic_fake)
            c_eps = self.epsilon_penalty(1e-3, critic_real)
            c_loss += c_wass + c_eps
            if self.use_gp:
                c_gp = self.gradient_penalty(real_data, fake_data, mask)
                c_loss += c_gp

            #update aa grids
            fake_atom_grid = torch.where(repl[:,:,None,None,None], fake_atom_grid, fake_atom)
            real_atom_grid = torch.where(repl[:,:,None,None,None], real_atom_grid, target_atom[:, None, :, :, :])

        self.opt_critic.zero_grad()
        c_loss.backward()
        self.opt_critic.step()

        return c_loss

    def train_step_gen(self, elems, initial, energy_ndx):

        aa_grid, cg_features = initial

        g_wass = torch.zeros([], dtype=torch.float32, device=self.device)

        fake_atom_grid = aa_grid.clone()
        real_atom_grid = aa_grid.clone()

        for target_atom, target_type, aa_featvec, repl, mask in elems:
            #prepare input for generator
            fake_aa_features = self.featurize(fake_atom_grid, aa_featvec)
            c_fake = fake_aa_features + cg_features
            z = torch.empty(
                [target_atom.shape[0], self.z_dim],
                dtype=torch.float32,
                device=self.device,
            ).normal_()

            #generate fake atom
            fake_atom = self.generator(z, target_type, c_fake)

            #critic
            critic_fake = self.critic(torch.cat([fake_atom, c_fake], dim=1))

            #mask
            critic_fake = torch.squeeze(critic_fake) * mask

            #loss
            g_wass += self.generator_loss(critic_fake)
            #g_loss += g_wass


            #update aa grids
            fake_atom_grid = torch.where(repl[:,:,None,None,None], fake_atom_grid, fake_atom)
            real_atom_grid = torch.where(repl[:,:,None,None,None], real_atom_grid, target_atom[:, None, :, :, :])

        #b1, a1, d1, l1 = self.get_energies_from_coords(real_coords, energy_ndx)
        #b2, a2, d2, l2 = self.get_energies_from_grid(real_atom_grid, energy_ndx)
        #print(a1,a2)
        #print(d1,d2)
        #print(l1,l2)

        if self.energy_prior_mode == 1:
            b_loss, a_loss, d_loss, l_loss, b_energy, a_energy, d_energy, l_energy = self.energy_loss_mean_abs_dif(real_atom_grid, fake_atom_grid, energy_ndx)
            energy_loss = self.covalent_weight*(b_loss + a_loss + d_loss) + self.lj_weight * l_loss
        elif self.energy_prior_mode == 2:
            energy_loss, b_energy, a_energy, d_energy, l_energy = self.free_energy_perturbation_loss(real_atom_grid, fake_atom_grid, energy_ndx, temp=568)
        elif self.energy_prior_mode == 3:
            energy_loss, b_energy, a_energy, d_energy, l_energy = self.force_matching_loss(real_coords, fake_atom_grid, energy_ndx)
        elif self.energy_prior_mode == 4:
            b_loss, a_loss, d_loss, l_loss, b_energy, a_energy, d_energy, l_energy = self.energy_loss_mean_sq_dif(real_atom_grid, fake_atom_grid, energy_ndx)
            energy_loss = self.covalent_weight*(b_loss + a_loss + d_loss) + self.lj_weight * l_loss
        elif self.energy_prior_mode == 5:
            b_loss, a_loss, d_loss, l_loss, b_energy, a_energy, d_energy, l_energy = self.truncated_energy_min_loss(real_atom_grid, fake_atom_grid, energy_ndx)
            energy_loss = self.covalent_weight*(b_loss + a_loss + d_loss) + self.lj_weight * l_loss
        else:
            b_energy, a_energy, d_energy, l_energy = self.energy_min_loss(fake_atom_grid, energy_ndx)

            energy_loss = self.covalent_weight*(b_energy + a_energy + d_energy) + self.lj_weight * l_energy

        g_loss = g_wass + self.prior_weights[self.step] * energy_loss
        #g_loss = g_wass

        self.opt_generator.zero_grad()
        g_loss.backward()
        self.opt_generator.step()

        return g_wass, energy_loss, b_energy, a_energy, d_energy, l_energy

    def val_step_gen(self, initial_atom_grid, bead_features, elems, energy_ndx, real_coords):

        g_wass = torch.zeros([], dtype=torch.float32, device=self.device)

        fake_atom_grid = initial_atom_grid.clone()
        real_atom_grid = initial_atom_grid.clone()

        for target_atom, target_type, aa_featvec, repl, mask in elems:
            #prepare input for generator
            fake_aa_features = self.featurize(fake_atom_grid, aa_featvec)
            c_fake = fake_aa_features + bead_features
            z = torch.empty(
                [target_atom.shape[0], self.z_dim],
                dtype=torch.float32,
                device=self.device,
            ).normal_()

            #generate fake atom
            fake_atom = self.generator(z, target_type, c_fake)

            #critic
            critic_fake = self.critic(torch.cat([fake_atom, c_fake], dim=1))

            #mask
            critic_fake = torch.squeeze(critic_fake) * mask

            #loss
            g_wass += self.generator_loss(critic_fake)
            #g_loss += g_wass


            #update aa grids
            fake_atom_grid = torch.where(repl[:,:,None,None,None], fake_atom_grid, fake_atom)
            real_atom_grid = torch.where(repl[:,:,None,None,None], real_atom_grid, target_atom[:, None, :, :, :])

        #b1, a1, d1, l1 = self.get_energies_from_coords(real_coords, energy_ndx)
        #b2, a2, d2, l2 = self.get_energies_from_grid(real_atom_grid, energy_ndx)
        #print(a1,a2)
        #print(d1,d2)
        #print(l1,l2)

        if self.energy_prior_mode == 1:
            b_loss, a_loss, d_loss, l_loss, b_energy, a_energy, d_energy, l_energy = self.energy_loss_mean_abs_dif(real_atom_grid, fake_atom_grid, energy_ndx)
            energy_loss = self.covalent_weight*(b_loss + a_loss + d_loss) + self.lj_weight * l_loss
        elif self.energy_prior_mode == 2:
            energy_loss, b_energy, a_energy, d_energy, l_energy = self.free_energy_perturbation_loss(real_atom_grid, fake_atom_grid, energy_ndx, temp=568)
        elif self.energy_prior_mode == 3:
            energy_loss, b_energy, a_energy, d_energy, l_energy = self.force_matching_loss(real_coords, fake_atom_grid, energy_ndx)
        elif self.energy_prior_mode == 4:
            b_loss, a_loss, d_loss, l_loss, b_energy, a_energy, d_energy, l_energy = self.energy_loss_mean_sq_dif(real_atom_grid, fake_atom_grid, energy_ndx)
            energy_loss = self.covalent_weight*(b_loss + a_loss + d_loss) + self.lj_weight * l_loss
        elif self.energy_prior_mode == 5:
            b_loss, a_loss, d_loss, l_loss, b_energy, a_energy, d_energy, l_energy = self.truncated_energy_min_loss(real_atom_grid, fake_atom_grid, energy_ndx)
            energy_loss = self.covalent_weight*(b_loss + a_loss + d_loss) + self.lj_weight * l_loss
        else:
            b_energy, a_energy, d_energy, l_energy = self.energy_min_loss(fake_atom_grid, energy_ndx)

            energy_loss = self.covalent_weight*(b_energy + a_energy + d_energy) + self.lj_weight * l_energy

        g_loss = g_wass + self.prior_weights[self.step] * energy_loss
        #g_loss = g_wass

        return g_wass, energy_loss, b_energy, a_energy, d_energy, l_energy

    def energy_test(self, initial_atom_grid, bead_features, elems, energy_ndx, aa_pos):
        real_atom_grid = initial_atom_grid.clone()

        #for target_atom, target_type, aa_featvec, repl, mask in elems:
            #prepare input for generator
            #real_atom_grid = torch.where(repl[:,:,None,None,None], real_atom_grid, target_atom[:, None, :, :, :])

        #testing
        b,a,d,lj = energy_ndx
        coords = avg_blob(
            real_atom_grid,
            res=self.cfg.getint('grid', 'resolution'),
            width=self.cfg.getfloat('grid', 'length'),
            sigma=self.cfg.getfloat('grid', 'sigma'),
            device=self.device,
        )
        torch.set_printoptions(threshold=5000)

        b_energy = torch.sum(bond_energy(coords, b, self.bond_params))
        a_energy = torch.sum(angle_energy(coords, a, self.angle_params))
        d_energy = torch.sum(dih_energy(coords, d, self.dih_params))
        l_energy = torch.sum(lj_energy(coords, lj, self.lj_params))

        return b_energy, a_energy, d_energy , l_energy

    def to_tensor_and_zip(self, *args):
        args = tuple(torch.from_numpy(x).float().to(self.device) if x.dtype == np.dtype(np.float64) else torch.from_numpy(x).to(self.device) for x in args)
        #args = tuple(torch.transpose(x, 0, 1) for x in args)
        elems = zip(*args)
        return elems

    def predict(self, fake_atom_grid, bead_features, elems, energy_ndx):
        new_atoms = []
        for target_type, aa_featvec, repl in elems:
            #prepare input for generator
            fake_aa_features = self.featurize(fake_atom_grid, aa_featvec)
            c_fake = fake_aa_features + bead_features
            z = torch.empty(
                [target_type.shape[0], self.z_dim],
                dtype=torch.float32,
                device=self.device,
            ).normal_()

            #generate fake atom
            fake_atom = self.generator(z, target_type, c_fake)
            new_atoms.append(fake_atom)

            #update aa grids
            fake_atom_grid = torch.where(repl[:,:,None,None,None], fake_atom_grid, fake_atom)

        new_atoms = torch.stack(new_atoms, dim = 1)
        new_atom_coords = avg_blob(
            new_atoms,
            res=self.cfg.getint('grid', 'resolution'),
            width=self.cfg.getfloat('grid', 'length'),
            sigma=self.cfg.getfloat('grid', 'sigma'),
            device=self.device,
        )

        coords = avg_blob(
            fake_atom_grid,
            res=self.cfg.getint('grid', 'resolution'),
            width=self.cfg.getfloat('grid', 'length'),
            sigma=self.cfg.getfloat('grid', 'sigma'),
            device=self.device,
        )

        bond_ndx, angle_ndx, dih_ndx, lj_ndx = energy_ndx
        b_energy = self.energy.bond(coords, bond_ndx)
        a_energy = self.energy.angle(coords, angle_ndx)
        d_energy = self.energy.dih(coords, dih_ndx)
        l_energy = self.energy.lj(coords, lj_ndx)

        return new_atom_coords, b_energy , a_energy , d_energy , l_energy

    def validate(self, samples_dir=None):

        mode = self.cfg.get('validate', 'bm_mode')

        if samples_dir:
            samples_dir = self.out.output_dir / samples_dir
            make_dir(samples_dir)
        else:
            samples_dir = self.out.samples_dir
        print("Saving samples in {}".format(samples_dir), "...", end='')

        resolution = self.cfg.getint('grid', 'resolution')
        delta_s = self.cfg.getfloat('grid', 'length') / self.cfg.getint('grid', 'resolution')
        sigma = self.cfg.getfloat('grid', 'sigma')
        grid = make_grid_np(delta_s, resolution)

        cg_kick = self.cfg.getfloat('universe', 'cg_kick')

        start = timer()

        #if samples_dir:
        #    self.samples_dir = self.model_dir + '/' + samples_dir
        #    self.make_dir(self.samples_dir)
        #print("Saving samples in {}".format(self.samples_dir), "...", end='')
        start_glob = timer()
        time = timer()
        #u_bm = deepcopy(self.u_val)
        bm_iter = self.val_data.make_recurrent_batch(bs=self.bs, train=False, mode="init", cg_kick=cg_kick)
        try:
            self.generator.eval()
            self.critic.eval()

            for batch in bm_iter:
                print("loop: ", timer()-time)
                time = timer()

                features, target_type, atom_featvec, repl, atom_pos, bead_pos, bead_featvec, b_ndx, a_ndx, d_ndx, lj_ndx = batch

                with torch.no_grad():
                    start = timer()
                    atom_grid = torch.from_numpy(voxelize_gauss(atom_pos, sigma, grid)).float().to(self.device)
                    bead_grid = voxelize_gauss(bead_pos, sigma, grid)
                    cg_features = bead_featvec[..., None, None, None] * bead_grid[:, :, None, :, :, :]
                    cg_features = torch.from_numpy(np.sum(cg_features, 1)).float().to(self.device)

                    repl = repl.astype(np.bool)
                    energy_ndx = (b_ndx, a_ndx, d_ndx, lj_ndx)
                    energy_ndx = tuple(torch.from_numpy(x).to(self.device) for x in energy_ndx)

                    elems = self.to_tensor_and_zip(target_type, atom_featvec, repl)

                    print("prep input: ", timer()-start)
                    start = timer()
                    new_coords, a,b,c,d = self.predict(atom_grid, cg_features, elems, energy_ndx)
                    print("predit: ", timer()- start)
                    start = timer()
                    new_coords = np.squeeze(new_coords)

                    if mode == "steep":
                        energies = a + b + c + d
                        energies = np.squeeze(energies)
                        ndx = energies.argmin()
                    elif mode == "lj_steep":
                        energies = d
                        energies = np.squeeze(energies)
                        ndx = energies.argmin()
                    elif mode == "bonded_steep":
                        energies = a + b + c
                        energies = np.squeeze(energies)
                        ndx = energies.argmin()
                    else:
                        ndx = np.random.randint(self.bs)


                    new_coords = new_coords[ndx, :, :].detach().cpu().numpy()
                    rot_mat = rot_mat_z(np.pi*2*ndx/self.bs)
                    new_coords = np.dot(new_coords, rot_mat.T)
                    for c, f in zip(new_coords, features):
                        f.atom.pos = f.rot_back(c)
                    print("insert new coords: ", timer()-start)
            print("total: ", timer()-start_glob)
            start = timer()
            for s in self.val_data.samples:
                f_name = s.name+"_init_" + str(self.step) + ".gro"
                s.write_gro_file(samples_dir / f_name)
            self.val_data.evaluate(folder=str(samples_dir)+"/", tag="_init_" + mode + "_" + str(self.step), ref=True)
            print("evaluate: ", timer()-start)

            print("init done!!!")

            for n in range(0, self.n_gibbs):
                bm_iter = self.val_data.make_recurrent_batch(bs=self.bs, train=False, mode="gibbs")
                for batch in bm_iter:
                    features, target_type, atom_featvec, repl, atom_pos, bead_pos, bead_featvec, b_ndx, a_ndx, d_ndx, lj_ndx = batch

                    with torch.no_grad():

                        atom_grid = torch.from_numpy(voxelize_gauss(atom_pos, sigma, grid)).float().to(self.device)
                        bead_grid = voxelize_gauss(bead_pos, sigma, grid)
                        cg_features = bead_featvec[..., None, None, None] * bead_grid[:, :, None, :, :, :]
                        cg_features = torch.from_numpy(np.sum(cg_features, 1)).float().to(self.device)


                        repl = repl.astype(np.bool)
                        energy_ndx = (b_ndx, a_ndx, d_ndx, lj_ndx)
                        energy_ndx = tuple(torch.from_numpy(x).to(self.device) for x in energy_ndx)

                        elems = self.to_tensor_and_zip(target_type, atom_featvec, repl)

                        new_coords, a, b, c, d = self.predict(atom_grid, cg_features, elems, energy_ndx)
                        new_coords = np.squeeze(new_coords)

                        if mode == "steep":
                            energies = a + b + c + d
                            energies = np.squeeze(energies)
                            ndx = energies.argmin()
                        elif mode == "lj_steep":
                            energies = d
                            energies = np.squeeze(energies)
                            ndx = energies.argmin()
                        elif mode == "bonded_steep":
                            energies = a + b + c
                            energies = np.squeeze(energies)
                            ndx = energies.argmin()
                        else:
                            ndx = np.random.randint(self.bs)

                        new_coords = new_coords[ndx, :, :].detach().cpu().numpy()
                        rot_mat = rot_mat_z(np.pi * 2 * ndx / self.bs)
                        new_coords = np.dot(new_coords, rot_mat.T)
                        for c, f in zip(new_coords, features):
                            f.atom.pos = f.rot_back(c)

                for s in self.val_data.samples:
                    f_name = s.name+"_gibbs"+str(n)+"_" + str(self.step) + ".gro"
                    s.write_gro_file(samples_dir / f_name)
                self.val_data.evaluate(folder=str(samples_dir)+"/", tag="_gibbs_" + mode + "_" + str(n) + "_" + str(self.step), ref=True)

            #reset atom positions
            self.val_data.kick_atoms()

            end = timer()
            print("done!", "time:", end - start)
        finally:
            self.generator.train()
            self.critic.train()

def rot_mat_z(theta):
    #rotation axis
    v_rot = np.array([0.0, 0.0, 1.0])

    #rotation angle
    #theta = np.random.uniform(0, np.pi * 2)

    #rotation matrix
    a = math.cos(theta / 2.0)
    b, c, d = -v_rot * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    rot_mat = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

    return rot_mat

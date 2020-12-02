import torch
from torch.optim import Adam, RMSprop
from torch.utils.data import Dataset, DataLoader
from torch_util import make_grid_np, rand_rotation_matrix, voxelize_gauss
import numpy as np
#from tqdm import tqdm
from timeit import default_timer as timer
import os
import math
#from configparser import ConfigParser
#import mdtraj as md
#from universe import *
import dbm.model as model
import dbm.tf_utils as tf_utils
import dbm.tf_energy as tf_energy
from copy import deepcopy
from shutil import copyfile
from contextlib import redirect_stdout
from operator import add

#tf.compat.v1.disable_eager_execution()
"""
# ## Read config
cfg = ConfigParser()
cfg.read('config.ini')

batchsize = cfg.getint('model', 'batchsize')
z_dim = int(cfg.getint('model', 'noise_dim'))
grid_size = cfg.getint('grid', 'max_resolution')
ds = float(cfg.getfloat('grid', 'length')) / float(cfg.getint('grid', 'max_resolution'))
sigma = cfg.getfloat('grid', 'sigma')

n_steps = cfg.getint('training', 'max_iters')
n_critic = cfg.getint('training', 'n_critic')
"""

class DS(Dataset):
    def __init__(self, data, sigma, resolution, delta_s, train=True, rand_rot=True):
        self.data = data
        self.sigma = sigma
        self.resolution = resolution
        self.delta_s = delta_s
        self.elems = list(self.data.recurrent_generator_combined(train=train, rand_rot=False))
        self.rand_rot = rand_rot
        self.grid = make_grid_np(self.delta_s, self.resolution)

    def __len__(self):
        return len(self.elems)

    def __getitem__(self, ndx):
        if self.rand_rot:
            R = self.rand_rot_mat(self.data.align)
        else:
            R = np.eye(3)

        target_pos, target_type, aa_feat, repl, mask, aa_pos, cg_pos, cg_feat, bonds_ndx, angles_ndx, dihs_ndx, ljs_ndx = self.elems[ndx]

        target_atom = voxelize_gauss(np.dot(target_pos, R.T), self.sigma, self.grid)
        atom_grid = voxelize_gauss(np.dot(aa_pos, R.T), self.sigma, self.grid)
        bead_grid = voxelize_gauss(np.dot(cg_pos, R.T), self.sigma, self.grid)

        cg_feat = cg_feat[None, None, None, :, :]
        bead_grid = bead_grid[:,:,:,:, None]
        bead_grid = cg_feat * bead_grid

        energy_ndx = (bonds_ndx, angles_ndx, dihs_ndx, ljs_ndx)

        return atom_grid, bead_grid, target_atom, target_type, aa_feat, repl, mask, energy_ndx

    def rand_rot_mat(self, align):
        #rotation axis
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

class GAN_SEQ():

    def __init__(self, name, model_type, u_train, u_val, z_dim, bs, sigma, resolution, delta_s, rand_rot):

        #config = tf.compat.v1.ConfigProto()
        #config.gpu_options.allow_growth = True
        #session = tf.compat.v1.Session(config=config)

        self.name = name

        self.z_dim = z_dim
        #self.sigma = tf.convert_to_tensor(sigma, tf.float32)
        #self.resolution = tf.convert_to_tensor(resolution, tf.float32)
        #self.delta_s = tf.convert_to_tensor(delta_s, tf.float32)
        self.sigma = sigma
        self.resolution = resolution
        self.delta_s = delta_s

        self.rand_rot = rand_rot

        self.bs = bs

        # Make Dirs for saving
        self.model_dir = './' + self.name
        self.checkpoint_dir = self.model_dir + '/checkpoints'
        self.samples_dir = self.model_dir + '/samples'
        self.logs_dir = self.model_dir + '/logs'
        self.make_dir(self.model_dir)
        self.make_dir(self.checkpoint_dir)
        self.make_dir(self.samples_dir)
        self.make_dir(self.logs_dir)

        print("Model dir:", self.model_dir)
        print("Checkpoint dir:", self.checkpoint_dir)
        print("Samples dir:", self.samples_dir)
        print("Logs dir:", self.logs_dir)

        #Data pipeline
        self.u_train = u_train
        self.u_val = u_val

        

        data_generator = lambda: self.u_train.recurrent_generator_combined(train=True, rand_rot=rand_rot)
        #data_generator = lambda: self.u_train.recurrent_generator(train=True, mode="init", rand_rot=rand_rot)

        ds_train = tf.data.Dataset.from_generator(data_generator, output_types=(tf.float32,   # target position
                                                                                tf.float32,   # target type
                                                                                tf.float32,   # env atoms featvec
                                                                                tf.bool,      # replace vector
                                                                                tf.float32,   # mask
                                                                                tf.float32,   # initial env atoms positions
                                                                                tf.float32,   # env beads positions
                                                                                tf.float32,   # env beads featvec
                                                                                tf.int32,     # bond indices
                                                                                tf.int32,     # angle indices
                                                                                tf.int32,     # dih indices
                                                                                tf.int32))    # lj indices


        ds_train = ds_train.batch(bs, drop_remainder=True)
        ds_train = ds_train.map(lambda t_pos, t_type, aa_feat, repl, mask, aa_pos, cg_pos, cg_feat, b_ndx, a_ndx, d_ndx, lj_ndx: (
            (tf.transpose(t_pos, [1, 0, 2, 3]),     # (Seq_len, BS, 1, dim)
            tf.transpose(t_type, [1, 0, 2, 3]),     # (Seq_len, BS, 1, dim)
            tf.transpose(aa_feat, [1, 0, 2, 3]),    # (Seq_len, BS, n_atoms, n_features)
            tf.transpose(repl, [1, 0, 2, 3, 4, 5]), # (Seq_len, BS, 1, 1, 1, n_atoms)
            tf.transpose(mask, [1, 0])),             # (Seq_len, BS)
            tf_utils.prepare_initial(aa_pos, cg_pos, cg_feat, self.sigma, self.resolution, self.delta_s),
            # 2*(BS, grid_dim, grid_dim, grid_dim, n_atoms), (BS, grid_dim, grid_dim, grid_dim, n_beads)
            (b_ndx,     # (BS, 3, n_bonds)
             a_ndx,     # (BS, 4, n_angles)
             d_ndx,     # (BS, 5, n_dihs)
             lj_ndx)))  # (BS, 3, n_ljs)
        ds_train = ds_train.repeat()
        ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

        self.ds_iter = iter(ds_train)

        self.n_atom_chns = u_train.ff.n_atom_chns

        #forcefield for energy loss
        self.ff = self.u_train.ff
        self.bond_params = tf.constant(self.ff.bond_params(), tf.float32)
        self.angle_params = tf.constant(self.ff.angle_params(), tf.float32)
        self.dih_params = tf.constant(self.ff.dih_params(), tf.float32)
        self.lj_params = tf.constant(self.ff.lj_params(), tf.float32)

        self.atom_mass = tf.constant([[[atype.mass for atype in self.ff.atom_types.values()]]]) #(1, 1, n_atomtypes)

        #Model selection
        if model_type == "Deep_g8":
            self.critic = model.Critic_8(name="Critic")
            self.generator = model.Generator_8(self.n_atom_chns, self.z_dim, name="Generator")
        elif model_type == "Deep_g8d":
            self.critic = model.Critic_8d(name="Critic")
            self.generator = model.Generator_8(self.n_atom_chns, self.z_dim, name="Generator")
        else:
            self.critic = model.Critic_8(name="Critic")
            self.generator = model.Generator_8(self.n_atom_chns, self.z_dim, name="Generator")

        #Optimizer
        self.gen_opt = tf.keras.optimizers.Adam(0.00005, beta_1=.0, beta_2=.9)
        #self.crit_opt = tf.keras.optimizers.Adam(0.0001, beta_1=.0, beta_2=.9)
        self.crit_opt = tf.keras.optimizers.Adam(0.0001, beta_1=.0, beta_2=.9)

        #Summaries and checkpoints
        self.summary_writer = tf.summary.create_file_writer(self.logs_dir)
        self.checkpoint_critic = tf.train.Checkpoint(optimizer=self.crit_opt, model=self.critic)
        self.checkpoint_manager_critic = tf.train.CheckpointManager(self.checkpoint_critic, self.checkpoint_dir+"/crit", max_to_keep=2)
        self.checkpoint_gen = tf.train.Checkpoint(optimizer=self.gen_opt, model=self.generator, step=tf.Variable(1))
        self.checkpoint_manager_gen = tf.train.CheckpointManager(self.checkpoint_gen, self.checkpoint_dir+"/gen", max_to_keep=2)

        #loss dict
        self.loss_dict = {
            "critic_tot": [],
            "critic_wass": [],
            "critic_gp": [],
            "critic_eps": [],
            "gen_tot": [],
            "gen_wass": [],
            "gen_com": [],
            "gen_bond": [],
            "gen_angle": [],
            "gen_dih": [],
            "gen_lj": []
        }

        #Restore old checkpoint if available
        status_critic = self.checkpoint_critic.restore(self.checkpoint_manager_critic.latest_checkpoint)
        status_gen = self.checkpoint_gen.restore(self.checkpoint_manager_gen.latest_checkpoint)
        if self.checkpoint_manager_gen.latest_checkpoint:
            print("Restored Generator from {}".format(self.checkpoint_manager_gen.latest_checkpoint))
        else:
            print("Initializing Generator from scratch.")
        if self.checkpoint_manager_critic.latest_checkpoint:
            print("Restored Critic from {}".format(self.checkpoint_manager_critic.latest_checkpoint))
        else:
            print("Initializing Critic from scratch.")

        #Metrics for log
        self.c_metrics = [('critic/loss', tf.keras.metrics.Mean()), ('critic/wass', tf.keras.metrics.Mean()),
                     ('critic/grad', tf.keras.metrics.Mean()), ('critic/eps', tf.keras.metrics.Mean())]
        self.g_metrics = [('generator/tot_loss', tf.keras.metrics.Mean()),
                          ('generator/w_loss', tf.keras.metrics.Mean()),
                          ('generator/b_loss', tf.keras.metrics.Mean()),
                          ('generator/a_loss', tf.keras.metrics.Mean()),
                          ('generator/d_loss', tf.keras.metrics.Mean()),
                          ('generator/lj_loss', tf.keras.metrics.Mean())]


    def train(self, prior_type, prior_start, prior_end, n_start_prior, n_fade_in_prior, n_steps, n_steps_pre, n_critic, n_tensorboard, n_save, n_val, n_gibbs, bm_mode):

        print("Training for n_steps={}, n_critic={}".format(n_steps, n_critic))
        with open('./' + self.name + '/config.ini', 'a') as f:
            f.write("# started at step: " + str(self.checkpoint_gen.step.numpy()))

        self.loss_dict["start_step"] = int(self.checkpoint_gen.step)

        if int(self.checkpoint_gen.step) == 1:
            print("pretraining...")
            for step in range(1, n_steps_pre+1):
                elems, initial, energy_ndx = next(self.ds_iter)
                g_loss_super, b_loss, a_loss, d_loss, l_loss = self.train_step_generator_supervised(elems, initial, energy_ndx)
                c_loss, c_loss_wass, c_loss_grad, c_loss_eps = self.train_step_critic(elems, initial)
                print("step ", step, g_loss_super.numpy(), b_loss.numpy(), a_loss.numpy(), d_loss.numpy(), l_loss.numpy(), "C: ", c_loss.numpy(), c_loss_wass.numpy(), c_loss_grad.numpy(), c_loss_eps.numpy())


        #self.validate(n_gibbs, bm_mode=bm_mode)

        prior_weight = tf.Variable(0.0)

        for step in range(1, n_steps + 1):

            # train critic
            start = timer()
            for i_critic in range(n_critic):
                elems, initial, _ = next(self.ds_iter)
                c_loss, c_loss_wass, c_loss_grad, c_loss_eps = self.train_step_critic(elems, initial)

                for (_, metric), loss in zip(self.c_metrics, [c_loss, c_loss_wass, c_loss_grad, c_loss_eps]):
                    metric(loss)



            # train generator
            if int(self.checkpoint_gen.step) >= n_start_prior:
                if int(self.checkpoint_gen.step) < n_start_prior + n_fade_in_prior:
                    prior_weight.assign(prior_end * (int(self.checkpoint_gen.step) - n_start_prior)/n_fade_in_prior)
                else:
                    prior_weight.assign(prior_end)
            else:
                prior_weight.assign(prior_start)
            elems, initial, energy_ndx = next(self.ds_iter)
            g_loss_tot, g_loss_w, g_loss_com, b_loss, a_loss, d_loss, l_loss = self.train_step_generator(elems, initial, energy_ndx, prior_weight, prior_type)
            for (_, metric), loss in zip(self.g_metrics, [g_loss_tot, g_loss_w, b_loss, a_loss, d_loss, l_loss]):
                metric(loss)


            end = timer()

            print(int(self.checkpoint_gen.step), "D: ", c_loss.numpy(), c_loss_wass.numpy(), c_loss_grad.numpy(), c_loss_eps.numpy())
            print(int(self.checkpoint_gen.step),"G: ",  g_loss_tot.numpy(), g_loss_w.numpy(), g_loss_com.numpy(), b_loss.numpy(), a_loss.numpy(), d_loss.numpy(), l_loss.numpy(), "time:", end - start)

            self.loss_dict["critic_tot"].append(c_loss.numpy())
            self.loss_dict["critic_wass"].append(c_loss_wass.numpy())
            self.loss_dict["critic_gp"].append(c_loss_grad.numpy())
            self.loss_dict["critic_eps"].append(c_loss_eps.numpy())
            self.loss_dict["gen_tot"].append(g_loss_tot.numpy())
            self.loss_dict["gen_wass"].append(g_loss_w.numpy())
            self.loss_dict["gen_com"].append(g_loss_com.numpy())
            self.loss_dict["gen_bond"].append(b_loss.numpy())
            self.loss_dict["gen_angle"].append(a_loss.numpy())
            self.loss_dict["gen_dih"].append(d_loss.numpy())
            self.loss_dict["gen_lj"].append(l_loss.numpy())

            self.checkpoint_gen.step.assign_add(1)


            if step % n_tensorboard == 0:
                with self.summary_writer.as_default():
                    for label, metric in self.c_metrics:
                        tf.summary.scalar(label, metric.result(), step=int(self.checkpoint_gen.step))
                        metric.reset_states()
                    for label, metric in self.g_metrics:
                        tf.summary.scalar(label, metric.result(), step=int(self.checkpoint_gen.step))
                        metric.reset_states()

            # record samples
            if step % n_val == 0:
                self.validate2(n_gibbs, bm_mode=bm_mode)

            # save model
            if step % n_save == 0:
                print("saving model")
                self.checkpoint_manager_critic.save()
                self.checkpoint_manager_gen.save()



        with open('./' + self.name + '/config.ini', 'a') as f:
            f.write("# ended at step: " + str(self.checkpoint_gen.step.numpy()))
        self.loss_dict["end_step"] = int(self.checkpoint_gen.step)-1
        print("saving loss dict")
        np.save(self.logs_dir+"/loss_dict_"+str(self.checkpoint_gen.step.numpy()), self.loss_dict)
        print("saving model")
        self.checkpoint_manager_critic.save()
        self.checkpoint_manager_gen.save()

    # Losses
    @tf.function
    def gen_loss_wass(self, critic_fake, mask):
        critic_fake = critic_fake * mask
        #tf.print("gen loss wass", critic_fake.shape)
        return tf.reduce_mean(-1. * critic_fake)

    @tf.function
    def gen_supervised_loss(self, real_frames, fake_frames):
        fine_frames_normed = tf.divide(real_frames, tf.reduce_sum(real_frames, axis=[1, 2, 3], keepdims=True))
        fake_frames_normed = tf.divide(fake_frames, tf.reduce_sum(fake_frames, axis=[1, 2, 3], keepdims=True))
        g_loss_super = tf.reduce_mean(tf.keras.losses.KLD(fine_frames_normed, fake_frames_normed))
        return g_loss_super

    @tf.function
    def gen_loss_energy(self, real_aa_grid, fake_aa_grid, energy_ndx):
        bond_ndx, angle_ndx, dih_ndx, lj_ndx = energy_ndx
        real_coords = tf_utils.average_blob_pos(real_aa_grid, self.resolution, self.delta_s)
        fake_coords = tf_utils.average_blob_pos(fake_aa_grid, self.resolution, self.delta_s)

        e_bond_real = tf_energy.bond_energy(real_coords, bond_ndx, self.bond_params)
        e_angle_real = tf_energy.angle_energy(real_coords, angle_ndx, self.angle_params)
        e_dih_real = tf_energy.dih_energy(real_coords, dih_ndx, self.dih_params)
        e_lj_real = tf_energy.lj_energy(real_coords, lj_ndx, self.lj_params)

        e_bond_fake = tf_energy.bond_energy2(fake_coords, bond_ndx, self.bond_params)
        e_angle_fake = tf_energy.angle_energy(fake_coords, angle_ndx, self.angle_params)
        e_dih_fake = tf_energy.dih_energy(fake_coords, dih_ndx, self.dih_params)
        e_lj_fake = tf_energy.lj_energy(fake_coords, lj_ndx, self.lj_params)

        bond_loss = tf.abs(e_bond_real - e_bond_fake)
        angle_loss = tf.abs(e_angle_real - e_angle_fake)
        dih_loss = tf.abs(e_dih_real - e_dih_fake)
        lj_loss = tf.abs(e_lj_real - e_lj_fake)

        return e_bond_fake, e_angle_fake, e_dih_fake, e_lj_fake
        #return e_bond_real, e_angle_real, e_dih_real, e_lj_real
        #return bond_loss, angle_loss, dih_loss, lj_loss


    @tf.function
    def crit_loss_wass(self, critic_real, critic_fake, mask):

        critic_real = critic_real * mask
        critic_fake = critic_fake * mask
        #tf.print("crit fake", critic_fake.shape)
        #tf.print("crit real", critic_real.shape)

        loss_on_generated = tf.reduce_mean(critic_fake)
        loss_on_real = tf.reduce_mean(critic_real)

        loss = loss_on_generated - loss_on_real
        return loss

    @tf.function
    def epsilon_penalty(self, epsilon, critic_real_outputs, mask):
        if epsilon > 0:
            penalties = tf.square(critic_real_outputs) * mask
            #tf.print("epsilon", penalties.shape)

            penalty = epsilon * tf.reduce_mean(penalties)
            return penalty
        return 0

    @tf.function
    def gradient_penalty(self, input_real, input_fake, mask, target=1.0, use_wgan_lp_loss=False):
        if input_real.shape.ndims is None:
            raise ValueError('`input_real` can\'t have unknown rank.')
        if input_fake.shape.ndims is None:
            raise ValueError('`input_fake` can\'t have unknown rank.')

        differences = input_real - input_fake
        batch_size = differences.shape[0] or tf.shape(differences)[0]
        alpha_shape = [batch_size] + [1] * (differences.shape.ndims - 1)
        alpha = tf.random.uniform(shape=alpha_shape, dtype=input_real.dtype)

        interpolates = input_real - (alpha * differences)
        critic_interpolates = self.critic(interpolates)

        gradients = tf.gradients(critic_interpolates, interpolates)[0]

        gradient_squares = tf.math.reduce_sum(tf.math.square(gradients), axis=list(range(1, gradients.shape.ndims)))
        # avoid annihilation in sum
        gradient_squares = tf.math.maximum(gradient_squares, 1e-5)
        # Propagate shape information, if possible.
        if isinstance(batch_size, int):
            gradient_squares.set_shape([batch_size] + gradient_squares.shape.as_list()[1:])

        # if check_numerics:
        #    gradient_squares = tf.debugging.check_numerics(gradient_squares, 'gradient_squares')

        gradient_norm = tf.math.sqrt(gradient_squares)
        # if check_numerics:
        #    gradient_norm = tf.debugging.check_numerics(gradient_norm, 'gradient_norm')

        if use_wgan_lp_loss:
            penalties = tf.square(tf.maximum(gradient_norm - target, 0)) / (target ** 2)
        else:
            penalties = tf.square(gradient_norm - target) / (target ** 2)

        penalty = penalties * mask
        #tf.print("gp", penalty.shape)
        penalty = tf.reduce_mean(penalty)

        return penalty

    @tf.function
    def iterate_gen_super(self, acc, elems):
        #Unpack accumulator and elements
        fake_aa_grid, real_aa_grid, cg_features, g_loss_wass = acc
        target_pos, target_type, aa_featvec, repl, mask = elems

        #prepare condition for generator
        fake_aa_features = fake_aa_grid[:, :, :, :, :, tf.newaxis]
        fake_aa_features = fake_aa_features * aa_featvec[:, tf.newaxis, tf.newaxis, tf.newaxis, :]
        fake_aa_features = tf.reduce_sum(fake_aa_features, axis=4)
        fake_c = fake_aa_features + cg_features

        #noise
        z = tf.random.normal([tf.shape(target_pos)[0], self.z_dim])

        #generate fake atom
        fake_atom = self.generator([z, fake_c, target_type])
        fake_atom = fake_atom * target_type[:, tf.newaxis, tf.newaxis, :, :]

        #update fake atom grid
        fake_atom_notype = tf.reduce_sum(fake_atom, axis=4, keepdims=True)
        fake_aa_grid = tf.where(repl, fake_aa_grid, fake_atom_notype)

        #update real aa grid
        target_atom = tf_utils.prepare_target(target_pos, target_type, self.sigma, self.resolution, self.delta_s)
        real_atom_notype = tf.reduce_sum(target_atom, axis=4, keepdims=True)
        real_aa_grid = tf.where(repl, real_aa_grid, real_atom_notype)

        g_loss_super = self.gen_supervised_loss(real_atom_notype, fake_atom_notype)

        #tf.print(repl, summarize=-1)

        return fake_aa_grid, real_aa_grid, cg_features, g_loss_super

    # ## Training steps
    @tf.function
    def iterate_gen(self, acc, elems):
        #Unpack accumulator and elements
        fake_aa_grid, real_aa_grid, cg_features, g_loss_wass, real_com, fake_com, mass = acc
        target_pos, target_type, aa_featvec, repl, mask = elems

        #prepare condition for generator
        fake_aa_features = fake_aa_grid[:, :, :, :, :, tf.newaxis]
        fake_aa_features = fake_aa_features * aa_featvec[:, tf.newaxis, tf.newaxis, tf.newaxis, :]
        fake_aa_features = tf.reduce_sum(fake_aa_features, axis=4)
        fake_c = fake_aa_features + cg_features

        #noise
        z = tf.random.normal([tf.shape(target_pos)[0], self.z_dim])

        #generate fake atom
        #tf.print(target_type)
        #target_type = tf.constant([[[0.0, 0.0, 0.0, 1.0]]])
        fake_atom = self.generator([z, fake_c, target_type])
        fake_atom = fake_atom * target_type[:, tf.newaxis, tf.newaxis, :, :]

        #update fake atom grid
        fake_atom_notype = tf.reduce_sum(fake_atom, axis=4, keepdims=True)
        fake_aa_grid = tf.where(repl, fake_aa_grid, fake_atom_notype)

        #update real aa grid
        target_atom = tf_utils.prepare_target(target_pos, target_type, self.sigma, self.resolution, self.delta_s)
        real_atom_notype = tf.reduce_sum(target_atom, axis=4, keepdims=True)
        real_aa_grid = tf.where(repl, real_aa_grid, real_atom_notype)

        #COM
        mass = tf.reduce_sum(self.atom_mass * target_type, axis=-1, keepdims=True)*mask[:, tf.newaxis, tf.newaxis] #(BS,1,n_atomtypes)
        #tf.print("mass", mass.shape)
        fake_pos = tf_utils.average_blob_pos(fake_atom_notype, self.resolution, self.delta_s) #(BS,1,3)
        fake_com = mass * fake_pos #* mask[:, tf.newaxis, tf.newaxis]
        real_com = mass * target_pos #* mask[:, tf.newaxis, tf.newaxis]


        #Gen Loss
        critic_input_fake = tf.concat([fake_atom, fake_c], axis=-1)
        critic_fake = tf.squeeze(self.critic(critic_input_fake, training=True))
        g_loss_wass = self.gen_loss_wass(critic_fake, mask)

        #tf.print(repl, summarize=-1)
        #fake_coord = tf_utils.average_blob_pos(fake_atom_notype, self.resolution, self.delta_s)
        #fake_dis = tf.square(fake_coord)
        #fake_dis = tf.reduce_sum(fake_dis, axis=-1)
        #fake_dis = tf.sqrt(fake_dis)
        #tf.print(fake_dis, summarize=-1)



        return fake_aa_grid, real_aa_grid, cg_features, g_loss_wass, real_com, fake_com, mass

    @tf.function
    def train_step_generator_supervised(self, elems, initial, energy_ndx):
        #add a term to the initial values to store loss
        initial += (tf.constant(0.0),)
        with tf.GradientTape() as gen_tape:
            #iterate over sequence (recurrent training)
            fake_aa_grid, real_aa_grid, _, g_loss_super = tf.scan(self.iterate_gen_super, elems, initial)

            bond_loss, angle_loss, dih_loss, lj_loss = self.gen_loss_energy(real_aa_grid[-1], fake_aa_grid[-1], energy_ndx)

            bond_loss = tf.reduce_mean(bond_loss)
            angle_loss = tf.reduce_mean(angle_loss)
            dih_loss = tf.reduce_mean(dih_loss)
            lj_loss = tf.reduce_mean(lj_loss)

            gen_loss = tf.reduce_sum(g_loss_super)

        #update weights
        gradients_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        #tf.print(gradients_generator)
        #gradients_generator = [tf.clip_by_value(grad, -1E8, 1E8) for grad in gradients_generator]
        self.gen_opt.apply_gradients(zip(gradients_generator, self.generator.trainable_variables))
        return gen_loss, bond_loss, angle_loss, dih_loss, lj_loss

    @tf.function
    def train_step_generator(self, elems, initial, energy_ndx, prior_weight, prior_type):
        #add a term to the initial values to store loss
        initial += (tf.constant(0.0), tf.zeros((self.bs, 1, 3)), tf.zeros((self.bs, 1, 3)), tf.zeros((self.bs, 1, 1)))
        with tf.GradientTape(persistent=True) as gen_tape:
            #iterate over sequence (recurrent training)
            fake_aa_grid, real_aa_grid, _, g_loss_wass, r_com, f_com, mass = tf.scan(self.iterate_gen, elems, initial)

            #bond_loss, angle_loss, dih_loss, lj_loss = self.gen_loss_energy(real_aa_grid[-1], fake_aa_grid[-1], energy_ndx)

            fake_coords = tf_utils.average_blob_pos(fake_aa_grid[-1], self.resolution, self.delta_s)
            real_coords = tf_utils.average_blob_pos(real_aa_grid[-1], self.resolution, self.delta_s)

            bond_ndx, angle_ndx, dih_ndx, lj_ndx = energy_ndx

            e_bond_real = tf_energy.bond_energy(real_coords, bond_ndx, self.bond_params)
            e_angle_real = tf_energy.angle_energy(real_coords, angle_ndx, self.angle_params)
            e_dih_real = tf_energy.dih_energy(real_coords, dih_ndx, self.dih_params)
            e_lj_real = tf_energy.lj_energy(real_coords, lj_ndx, self.lj_params)

            e_bond_fake = tf_energy.bond_energy(fake_coords, bond_ndx, self.bond_params)
            e_angle_fake = tf_energy.angle_energy(fake_coords, angle_ndx, self.angle_params)
            e_dih_fake = tf_energy.dih_energy(fake_coords, dih_ndx, self.dih_params)
            e_lj_fake = tf_energy.lj_energy(fake_coords, lj_ndx, self.lj_params)

            if prior_type == 1:
                bond_loss = tf.abs(e_bond_real - e_bond_fake)
                angle_loss = tf.abs(e_angle_real - e_angle_fake)
                dih_loss = tf.abs(e_dih_real - e_dih_fake)
                lj_loss = tf.abs(e_lj_real - e_lj_fake)

                bond_loss = tf.reduce_mean(bond_loss)
                angle_loss = tf.reduce_mean(angle_loss)
                dih_loss = tf.reduce_mean(dih_loss)
                lj_loss = tf.reduce_mean(lj_loss)

                e_loss = prior_weight*(bond_loss + angle_loss + dih_loss + lj_loss)
            elif prior_type == 2:
                bond_loss = tf.abs(e_bond_real - e_bond_fake)
                angle_loss = tf.abs(e_angle_real - e_angle_fake)
                dih_loss = tf.abs(e_dih_real - e_dih_fake)
                lj_loss = tf.abs(e_lj_real - e_lj_fake)

                bond_loss = tf.reduce_mean(bond_loss)
                angle_loss = tf.reduce_mean(angle_loss)
                dih_loss = tf.reduce_mean(dih_loss)
                lj_loss = tf.reduce_mean(lj_loss)

                e_loss = prior_weight*(bond_loss + angle_loss + dih_loss + 10*lj_loss)
            elif prior_type == 3:
                bond_loss = tf.reduce_mean(e_bond_fake)
                angle_loss = tf.reduce_mean(e_angle_fake)
                dih_loss = tf.reduce_mean(e_dih_fake)
                lj_loss = tf.reduce_mean(e_lj_fake)

                e_loss = prior_weight * (bond_loss + angle_loss + dih_loss + lj_loss)
            elif prior_type == 4:
                bond_loss = tf.reduce_mean(e_bond_fake)
                angle_loss = tf.reduce_mean(e_angle_fake)
                dih_loss = tf.reduce_mean(e_dih_fake)
                lj_loss = tf.reduce_mean(e_lj_fake)

                e_loss = prior_weight * (bond_loss + angle_loss + dih_loss + 10*lj_loss)
            elif prior_type == 5:
                bond_loss = tf.abs(e_bond_real - e_bond_fake)
                angle_loss = tf.abs(e_angle_real - e_angle_fake)
                dih_loss = tf.abs(e_dih_real - e_dih_fake)

                bond_loss = tf.reduce_mean(bond_loss)
                angle_loss = tf.reduce_mean(angle_loss)
                dih_loss = tf.reduce_mean(dih_loss)
                lj_loss = tf.reduce_mean(e_lj_fake)

                e_loss = prior_weight * (bond_loss + angle_loss + dih_loss + lj_loss)
            elif prior_type == 6:
                bond_loss = tf.abs(e_bond_real - e_bond_fake)
                angle_loss = tf.abs(e_angle_real - e_angle_fake)
                dih_loss = tf.abs(e_dih_real - e_dih_fake)

                bond_loss = tf.reduce_mean(bond_loss)
                angle_loss = tf.reduce_mean(angle_loss)
                dih_loss = tf.reduce_mean(dih_loss)
                lj_loss = tf.reduce_mean(e_lj_fake)

                e_loss = prior_weight * (bond_loss + angle_loss + dih_loss + 10 * lj_loss)

            g_loss_wass = tf.reduce_sum(g_loss_wass)
            #bond_loss = tf.reduce_mean(bond_loss2)
            #angle_loss = tf.reduce_mean(angle_loss)
            #dih_loss = tf.reduce_mean(dih_loss)
            #lj_loss = tf.reduce_mean(lj_loss)

            #(N_seq, BS,1,3)
            r_com = tf.reduce_sum(r_com, axis=0)/tf.reduce_sum(mass, axis=0)
            f_com = tf.reduce_sum(f_com, axis=0)/tf.reduce_sum(mass, axis=0)
            #tf.print(tf.reduce_sum(mass, axis=0), summarize=-1)

            com_loss = f_com - r_com
            com_loss = tf.square(com_loss)
            com_loss = tf.reduce_mean(com_loss)

            #gen_loss = g_loss_wass + com_loss + prior_weight*(bond_loss + angle_loss + dih_loss + lj_loss)
            #gen_loss = g_loss_wass + prior_weight*(bond_loss + angle_loss + dih_loss + lj_loss)
            #gen_loss = g_loss_wass + com_loss
            #e_loss = prior_weight * (bond_loss + angle_loss + dih_loss + tf.math.log(lj_loss))
            #e_loss = bond_loss + angle_loss + dih_loss + lj_loss

            tot_loss = g_loss_wass + com_loss + e_loss
            #tot_loss = g_loss_wass + com_loss + prior_weight*(bond_loss + angle_loss + dih_loss + lj_loss )  #+ prior_weight*bond_loss
            #tot_loss = bond_loss + angle_loss + dih_loss + tf.math.log(lj_loss)
            #tf.print(prior_weight)
            tot_loss = tf.debugging.check_numerics(tot_loss, "tot loss sucks")

            #for loss in bond_loss2:
            #    tf.print(gen_tape.gradient(loss, [self.generator.trainable_variables[0]])[0][0], loss)
        #update weights
        #adv_grads = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        #energy_grads = gen_tape.gradient(e_loss, self.generator.trainable_variables)

        grads = gen_tape.gradient(tot_loss, self.generator.trainable_variables)
        #tf.print(bond_loss2)


        l = list(range(0, len(grads)))
        grads = [tf.debugging.check_numerics(g, "grads sucks "+str(l)) for (g,l) in zip(grads, l)]
        #tf.print(energy_grads)
        #energy_grads = [tf.clip_by_value(grad, -1E8, 1E8) for grad in energy_grads]
        #energy_grads = [tf.clip_by_value(grad, -0.1, 0.1) for grad in energy_grads]
        #normed_adv_grads, _ = tf.clip_by_global_norm(adv_grads, 0.8)
        #normed_energy_grads, _ = tf.clip_by_global_norm(energy_grads, 0.05)
        #tot_grads = gradients_generator + energy_grads
        #tot_grads = list(map(add, adv_grads, energy_grads))
        #tot_grads = list(map(add, adv_grads, energy_grads))
        #tot_grads = [a+b for (a,b) in zip(adv_grads, energy_grads)]
        #tf.print(tot_grads)
        #tf.print(tot_grads.shape)

        #for e in adv_grads:
        #    tf.print("wass", e.shape)
        #for e in energy_grads:
        #    tf.print("en", e)
        #for e in tot_grads:
        #    tf.print("tot", e.shape)

        #tf.print(gradients_generator)
        #gradients_generator = [tf.clip_by_value(grad, -1E16, 1E16) for grad in gradients_generator]
        #self.gen_opt.apply_gradients(zip(gradients_generator, self.generator.trainable_variables))
        self.gen_opt.apply_gradients(zip(grads, self.generator.trainable_variables))

        return e_loss, g_loss_wass, com_loss, bond_loss, angle_loss, dih_loss, lj_loss

    @tf.function
    def iterate_critic(self, acc, elems):
        #Unpack accumulator and elements
        fake_aa_grid, real_aa_grid, cg_features, c_loss_wass, c_loss_grad, c_loss_eps = acc
        target_pos, target_type, aa_featvec, repl, mask = elems

        #prepare condition for generator
        fake_aa_features = fake_aa_grid[:, :, :, :, :, tf.newaxis]
        fake_aa_features = fake_aa_features * aa_featvec[:, tf.newaxis, tf.newaxis, tf.newaxis, :]
        fake_aa_features = tf.reduce_sum(fake_aa_features, axis=4)
        fake_c = fake_aa_features + cg_features

        #prepare condition for critic
        real_aa_features = real_aa_grid[:, :, :, :, :, tf.newaxis]
        real_aa_features = real_aa_features * aa_featvec[:, tf.newaxis, tf.newaxis, tf.newaxis, :]
        real_aa_features = tf.reduce_sum(real_aa_features, axis=4)
        real_c = real_aa_features + cg_features

        #noise
        z = tf.random.normal([tf.shape(target_pos)[0], self.z_dim])

        #generate fake atom
        fake_atom = self.generator([z, fake_c, target_type])
        fake_atom = fake_atom * target_type[:, tf.newaxis, tf.newaxis, :, :]

        #get target atom
        target_atom = tf_utils.prepare_target(target_pos, target_type, self.sigma, self.resolution, self.delta_s)

        #update fake aa grid
        fake_atom_notype = tf.reduce_sum(fake_atom, axis=4, keepdims=True)
        fake_aa_grid = tf.where(repl, fake_aa_grid, fake_atom_notype)

        #update real aa grid
        real_atom_notype = tf.reduce_sum(target_atom, axis=4, keepdims=True)
        real_aa_grid = tf.where(repl, real_aa_grid, real_atom_notype)

        # Critic Loss
        critic_input_fake = tf.concat([fake_atom, fake_c], axis=-1)
        critic_fake = tf.squeeze(self.critic(critic_input_fake, training=True))

        critic_input_real = tf.concat([target_atom, real_c], axis=-1)
        critic_real = tf.squeeze(self.critic(critic_input_real, training=True))

        c_loss_wass = self.crit_loss_wass(critic_real, critic_fake, mask)
        c_loss_grad = self.gradient_penalty(critic_input_real, critic_input_fake, mask)
        c_loss_eps = self.epsilon_penalty(1e-3, critic_real, mask)

        return fake_aa_grid, real_aa_grid, cg_features, c_loss_wass, c_loss_grad, c_loss_eps

    #somehow doesn't work with @tf.function ...
    #@tf.function
    def train_step_critic(self, elems, initial):
        #add a terms to the initial values to store losses
        initial += (tf.constant(0.0), tf.constant(0.0), tf.constant(0.0))
        with tf.GradientTape() as critic_tape:
            #iterate over sequence
            _, _, _, c_loss_wass, c_loss_grad, c_loss_eps = tf.scan(self.iterate_critic, elems, initial, back_prop=True)

            c_loss_wass = tf.reduce_sum(c_loss_wass)
            c_loss_grad = tf.reduce_sum(c_loss_grad)
            c_loss_eps = tf.reduce_sum(c_loss_eps)

            c_loss = c_loss_wass + c_loss_grad * 10.0 + c_loss_eps
            #c_loss =  c_loss_grad * 10.0


        #update weights
        gradients_critic = critic_tape.gradient(c_loss, self.critic.trainable_variables)
        self.crit_opt.apply_gradients(zip(gradients_critic, self.critic.trainable_variables))
        #self.apply_grads_critic(gradients_critic)

        return c_loss, c_loss_wass, c_loss_grad, c_loss_eps

    #@tf.function
    #def apply_grads_critic(self, gradients):
    #    self.crit_opt.apply_gradients(zip(gradients, self.critic.trainable_variables))



    # ## Training steps
    @tf.function
    def iterate_gen_predict(self, acc, elems):
        #Unpack accumulator and elements
        fake_aa_grid, cg_features, fake_pos = acc
        target_type, aa_featvec, repl = elems

        #prepare condition for generator
        fake_aa_features = fake_aa_grid[:, :, :, :, :, tf.newaxis]
        fake_aa_features = fake_aa_features * aa_featvec[:, tf.newaxis, tf.newaxis, tf.newaxis, :]
        fake_aa_features = tf.reduce_sum(fake_aa_features, axis=4)
        fake_c = fake_aa_features + cg_features

        #noise
        z = tf.random.normal([tf.shape(target_type)[0], self.z_dim])

        #generate fake atom
        fake_atom = self.generator([z, fake_c, target_type])
        fake_atom = fake_atom * target_type[:, tf.newaxis, tf.newaxis, :, :]

        #update fake atom grid
        fake_atom_notype = tf.reduce_sum(fake_atom, axis=4, keepdims=True)
        fake_aa_grid = tf.where(repl, fake_aa_grid, fake_atom_notype)
        fake_pos = tf_utils.average_blob_pos(fake_atom_notype, self.resolution, self.delta_s) #(BS,1,3)


        return fake_aa_grid, cg_features, fake_pos

    @tf.function
    def predict2(self, target_type, repl, atom_pos, atom_featvec, bead_pos, bead_featvec, bond_ndx, angle_ndx, dih_ndx, lj_ndx):
        print("Tracing predict!!!")

        aa_grid, cg_grid = tf_utils.prepare_initial_predict(atom_pos, bead_pos, bead_featvec, self.sigma, self.resolution,
                                         self.delta_s)
        initial = (aa_grid, cg_grid, tf.zeros((self.bs, 1, 3)))

        elems = (target_type, atom_featvec, repl)

        fake_aa_grid, _, new_coords = tf.scan(self.iterate_gen_predict, elems, initial)

        coords = tf_utils.average_blob_pos(fake_aa_grid[-1], self.resolution, self.delta_s)

        e_bond = tf_energy.bond_energy(coords, bond_ndx, self.bond_params)
        e_angle = tf_energy.angle_energy(coords, angle_ndx, self.angle_params)
        e_dih = tf_energy.dih_energy(coords, dih_ndx, self.dih_params)
        e_lj = tf_energy.lj_energy(coords, lj_ndx, self.lj_params)
        e = e_bond + e_angle + e_dih + e_lj
        #tf.print(e_bond)
        #tf.print(e_angle)
        #tf.print(e_dih)
        #tf.print(e_lj)
        return new_coords, e

    def validate2(self, n_gibbs, bm_mode="normal", samples_dir=None):
        start = timer()

        if samples_dir:
            self.samples_dir = self.model_dir + '/' + samples_dir
            self.make_dir(self.samples_dir)
        print("Saving samples in {}".format(self.samples_dir), "...", end='')

        #u_bm = deepcopy(self.u_val)
        bm_iter = self.u_val.make_recurrent_batch(bs=self.bs, train=False, mode="init")
        for batch in bm_iter:
            features, target_type, atom_featvec, repl, atom_pos, bead_pos, bead_featvec, b_ndx, a_ndx, d_ndx, lj_ndx = batch


            #print(energy_ndx.shape)
            #energy_ndx = tf.convert_to_tensor(energy_ndx)

            atom_featvec = tf.convert_to_tensor(atom_featvec, tf.float32)
            target_type = tf.convert_to_tensor(target_type, tf.float32)

            new_coords, energies = self.predict2(target_type, repl, atom_pos, atom_featvec, bead_pos, bead_featvec, b_ndx, a_ndx, d_ndx, lj_ndx)
            #print(energies)
            energies = np.squeeze(energies)
            new_coords = np.squeeze(new_coords)


            min_ndx = energies.argmin()
            new_coords = new_coords[:, min_ndx, :]
            rot_mat = self.rot_mat_z(np.pi*2*min_ndx/self.bs)
            new_coords = np.dot(new_coords, rot_mat.T)
            for c, f in zip(new_coords, features):
                f.atom.pos = f.rot_back(c)

        for u in self.u_val.samples:
            u.write_gro_file(self.samples_dir + "/"+u.name+"_init_" + str(self.checkpoint_gen.step.numpy()) + ".gro")
        self.u_val.evaluate(folder=self.samples_dir+"/", tag="_init_" + str(self.checkpoint_gen.step.numpy()), ref=True)

        print("init done!!!")

        for n in range(0, n_gibbs):
            bm_iter = self.u_val.make_recurrent_batch(bs=self.bs, train=False, mode="gibbs")
            for batch in bm_iter:
                features, target_type, atom_featvec, repl, atom_pos, bead_pos, bead_featvec, b_ndx, a_ndx, d_ndx, lj_ndx = batch
                atom_featvec = tf.convert_to_tensor(atom_featvec, tf.float32)
                target_type = tf.convert_to_tensor(target_type, tf.float32)

                new_coords, energies = self.predict2(target_type, repl, atom_pos, atom_featvec, bead_pos, bead_featvec, b_ndx, a_ndx, d_ndx, lj_ndx)
                #print(energies)
                energies = np.squeeze(energies)
                new_coords = np.squeeze(new_coords)

                #print(energies.shape)
                #print(new_coords.shape)

                min_ndx = energies.argmin()
                new_coords = new_coords[:, min_ndx, :]
                rot_mat = self.rot_mat_z(np.pi * 2 * min_ndx / self.bs)
                new_coords = np.dot(new_coords, rot_mat.T)
                for c, f in zip(new_coords, features):
                    f.atom.pos = f.rot_back(c)

            for u in self.u_val.samples:
                u.write_gro_file(self.samples_dir + "/"+u.name+"_gibbs"+str(n)+"_" + str(self.checkpoint_gen.step.numpy()) + ".gro")
            self.u_val.evaluate(folder=self.samples_dir+"/", tag="_gibbs_"+ str(n) + "_" + str(self.checkpoint_gen.step.numpy()), ref=True)

        #reset atom positions
        self.u_val.kick_atoms()

        end = timer()
        print("done!", "time:", end - start)

    def rot_mat_z(self, theta):
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


    @tf.function
    def predict(self, target_type, atom_pos, atom_featvec, bead_pos, bead_featvec):
        print("Tracing predict!!!")
        #tf.print(atom_pos.shape, atom_pos.dtype)
        #tf.print(atom_featvec.shape, atom_featvec.dtype)
        #tf.print(atom_featvec.shape, atom_featvec.dtype)
        #tf.print(bead_pos.shape, bead_pos.dtype)
        #tf.print(bead_featvec.shape, bead_featvec.dtype)
        #tf.print(self.sigma.shape, self.sigma.dtype)
        #tf.print(self.resolution.shape, self.resolution.dtype)
        #tf.print(self.delta_s.shape, self.delta_s.dtype)

        env_grid = tf_utils.prepare_grid(atom_pos, atom_featvec, bead_pos, bead_featvec, self.sigma, self.resolution,
                                         self.delta_s)
        z = tf.random.normal([tf.shape(env_grid)[0], self.z_dim])

        fake_atom = self.generator([z, env_grid, target_type], training=False)
        target_type = tf.cast(target_type, tf.float32)
        fake_atom_notype = fake_atom * target_type[:, tf.newaxis, tf.newaxis, :, :]
        fake_atom_notype = tf.reduce_sum(fake_atom_notype, axis=4, keepdims=True)
        new_pos = tf_utils.average_blob_pos(fake_atom_notype, self.resolution, self.delta_s)

        return new_pos

    #@tf.function
    def validate(self, n_gibbs, bm_mode="normal", samples_dir=None):
        start = timer()

        if samples_dir:
            self.samples_dir = self.model_dir + '/' + samples_dir
            self.make_dir(self.samples_dir)
        print("Saving samples in {}".format(self.samples_dir), "...", end='')

        #u_bm = deepcopy(self.u_val)
        bm_iter = self.u_val.make_batch(mode="init")
        for batch in bm_iter:
            features, target_type, atom_pos, atom_featvec, bead_pos, bead_featvec = batch
            if self.rand_rot:
                rot_mat = self.u_val.samples[0].rand_rot_mat()
                atom_pos = [np.dot(pos, rot_mat) for pos in atom_pos]
                bead_pos = [np.dot(pos, rot_mat) for pos in bead_pos]
            #target_type = tf.convert_to_tensor(target_type)
            #atom_pos = tf.convert_to_tensor(atom_pos)
            #atom_featvec = tf.convert_to_tensor(atom_featvec)
            #bead_pos = tf.convert_to_tensor(bead_pos)
            #bead_featvec = tf.convert_to_tensor(bead_featvec)
            new_pos = self.predict(target_type, atom_pos, atom_featvec, bead_pos, bead_featvec)
            new_pos = new_pos.numpy()[:,0,:]
            if self.rand_rot:
                new_pos = np.dot(new_pos, rot_mat.T)
            self.u_val.update_pos(features, new_pos, bm_mode="normal", temp=568)

        for u in self.u_val.samples:
            u.write_gro_file(self.samples_dir + "/"+u.name+"_init_" + str(self.checkpoint_gen.step.numpy()) + ".gro")
        self.u_val.evaluate(folder=self.samples_dir+"/", tag="_init_" + str(self.checkpoint_gen.step.numpy()), ref=True)

        print("init done!!!")

        for n in range(0, n_gibbs):
            bm_iter = self.u_val.make_batch(mode="gibbs")
            for batch in bm_iter:
                features, target_type, atom_pos, atom_featvec, bead_pos, bead_featvec = batch
                if self.rand_rot:
                    rot_mat = self.u_val.samples[0].rand_rot_mat()
                    atom_pos = [np.dot(pos, rot_mat) for pos in atom_pos]
                    bead_pos = [np.dot(pos, rot_mat) for pos in bead_pos]
                #target_type = tf.convert_to_tensor(target_type)
                #atom_pos = tf.convert_to_tensor(atom_pos)
                #atom_featvec = tf.convert_to_tensor(atom_featvec)
                #bead_pos = tf.convert_to_tensor(bead_pos)
                #bead_featvec = tf.convert_to_tensor(bead_featvec)
                new_pos = self.predict(target_type, atom_pos, atom_featvec, bead_pos, bead_featvec)
                new_pos = new_pos.numpy()[:,0,:]
                if self.rand_rot:
                    new_pos = np.dot(new_pos, rot_mat.T)
                self.u_val.update_pos(features, new_pos, bm_mode=bm_mode, temp=568)

            for u in self.u_val.samples:
                u.write_gro_file(self.samples_dir + "/"+u.name+"_gibbs"+str(n)+"_" + str(self.checkpoint_gen.step.numpy()) + ".gro")
            self.u_val.evaluate(folder=self.samples_dir+"/", tag="_gibbs_"+ str(n) + "_" + str(self.checkpoint_gen.step.numpy()), ref=True)

        #reset atom positions
        self.u_val.kick_atoms()

        end = timer()
        print("done!", "time:", end - start)



    def make_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

"""
u = Universe("./sPS_t568_small", align=True, aug=True)
#u = Universes(["./sPS_t568_small"], align=True, aug=True)
gan = GAN("test_model", "Deep_g8", u, u)
gan.train(n_steps, n_critic)
"""
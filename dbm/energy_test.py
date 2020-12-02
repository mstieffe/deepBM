import tensorflow as tf
import numpy as np
#from tqdm import tqdm
from timeit import default_timer as timer
import os
#from configparser import ConfigParser
#import mdtraj as md
#from universe import *
import dbm.model as model
import dbm.tf_utils as tf_utils
import dbm.tf_energy as tf_energy
from copy import deepcopy
from shutil import copyfile
from contextlib import redirect_stdout

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
class GAN_SEQ():

    def __init__(self, name, model_type, u_train, u_val, z_dim, bs, sigma, resolution, delta_s):

        self.name = name

        self.z_dim = z_dim
        self.sigma = sigma
        self.resolution = resolution
        self.delta_s = delta_s

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

        data_generator = lambda: self.u_train.recurrent_generator(train=True, mode="gibbs")

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
            aa_pos,             # (Seq_len, BS)
            #tf_utils.prepare_initial(aa_pos, cg_pos, cg_feat, self.sigma, self.resolution, self.delta_s),
            aa_pos,
            # 2*(BS, grid_dim, grid_dim, grid_dim, n_atoms), (BS, grid_dim, grid_dim, grid_dim, n_beads)
            (b_ndx,     # (BS, 3, n_bonds)
             a_ndx,     # (BS, 4, n_angles)
             d_ndx,     # (BS, 5, n_dihs)
             lj_ndx),   # (BS, 3, n_ljs)
            aa_pos))    # (BS, n_atoms, 3)
        #ds_train = ds_train.cache()
        ds_train = ds_train.repeat()
        ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

        self.ds_iter = iter(ds_train)

        self.n_atom_chns = u_train.ff.n_atom_chns

        self.ff = self.u_train.ff
        self.bond_params = tf.constant(self.ff.bond_params(), tf.float32)
        self.angle_params = tf.constant(self.ff.angle_params(), tf.float32)
        self.dih_params = tf.constant(self.ff.dih_params(), tf.float32)
        self.lj_params = tf.constant(self.ff.lj_params(), tf.float32)

        print(self.angle_params)
        print(self.dih_params)
        """
        #Model selection
        if model_type == "Deep_g8":
            self.critic = model.Critic_8(name="Critic")
            self.generator = model.Generator_8(self.n_atom_chns, self.z_dim, name="Generator")
        elif model_type == "Deep_g16":
            self.critic = model.Critic(name="Critic")
            self.generator = model.Generator(self.n_atom_chns, self.z_dim, name="Generator")
        else:
            self.critic = model.Critic_8(name="Critic")
            self.generator = model.Generator_8(self.n_atom_chns, self.z_dim, name="Generator")

        #Optimizer
        self.gen_opt = tf.keras.optimizers.Adam(0.00005, beta_1=.0, beta_2=.9)
        self.crit_opt = tf.keras.optimizers.Adam(0.0001, beta_1=.0, beta_2=.9)

        #Summaries and checkpoints
        self.summary_writer = tf.summary.create_file_writer(self.logs_dir)
        self.checkpoint_critic = tf.train.Checkpoint(optimizer=self.crit_opt, model=self.critic)
        self.checkpoint_manager_critic = tf.train.CheckpointManager(self.checkpoint_critic, self.checkpoint_dir, max_to_keep=2)
        self.checkpoint_gen = tf.train.Checkpoint(optimizer=self.gen_opt, model=self.generator, step=tf.Variable(1))
        self.checkpoint_manager_gen = tf.train.CheckpointManager(self.checkpoint_gen, self.checkpoint_dir, max_to_keep=2)

        #Restore old checkpoint if available
        status_critic = self.checkpoint_critic.restore(self.checkpoint_manager_critic.latest_checkpoint)
        status_gen = self.checkpoint_gen.restore(self.checkpoint_manager_gen.latest_checkpoint)
        if self.checkpoint_manager_gen.latest_checkpoint and self.checkpoint_manager_critic.latest_checkpoint:
            print("Restored from {}".format(self.checkpoint_manager_gen.latest_checkpoint))
            print("Restored from {}".format(self.checkpoint_manager_critic.latest_checkpoint))
        else:
            print("Initializing from scratch.")

        #Metrics for log
        self.c_metrics = [('critic/loss', tf.keras.metrics.Mean()), ('critic/wass', tf.keras.metrics.Mean()),
                     ('critic/grad', tf.keras.metrics.Mean()), ('critic/eps', tf.keras.metrics.Mean())]
        self.g_metrics = [('generator/loss', tf.keras.metrics.Mean())]
        """

    def train(self, n_steps, n_critic, n_tensorboard, n_save, n_val):

        print("Training for n_steps={}, n_critic={}".format(n_steps, n_critic))

        start= timer()
        tot_e1 = np.zeros(4)
        tot_e2 = np.zeros(4)
        tot_e3 = np.zeros(4)
        tot_e4 = np.zeros(4)

        #for step in range(5832):
        count = 0
        for elems, initial, energy_ndx, ref_coords in self.ds_iter:
            if count == 720:
                tot_e = np.zeros(4)
                print("it took ", timer() -start)
                start=timer()
                break
            e1,e2,e3,e4 = self.train_step_generator(elems, initial, energy_ndx, ref_coords)
            #print(e1.numpy())
            #e = np.zeros((32))
            #print(e.numpy())
            tot_e1 += e1.numpy()
            tot_e2 += e2.numpy()
            tot_e3 += e3.numpy()
            tot_e4 += e4.numpy()

            count += 1
            print(count)
            print("total energy: ", tot_e1, tot_e2, tot_e3, tot_e4)


        print("first took ", timer()-start)
        """
        start= timer()
        tot_e = np.zeros(4)
        #for step in range(5832):
        count = 0
        for elems, initial, energy_ndx, ref_coords in self.ds_iter:
            if count == 720:
                break
            e = self.train_step_generator(elems, initial, energy_ndx, ref_coords)
            #e = np.zeros((32))
            #print(e.numpy())
            #tot_e += np.sum(e.numpy())
            tot_e += e.numpy()

            count += 1
            print(count)
            print("total energy: ", tot_e)
        print("second took ", timer()-start)
        """
    # Losses
    @tf.function
    def gen_loss_wass(self, critic_fake, mask):
        critic_fake = critic_fake * mask
        return tf.reduce_mean(-1. * critic_fake)

    @tf.function
    def energy(self, ref_coords, energy_ndx):
        bond_ndx, angle_ndx, dih_ndx, lj_ndx = energy_ndx
        #real_coords = tf_utils.average_blob_pos(real_aa_grid, self.resolution, self.delta_s)
        #fake_coords = tf_utils.average_blob_pos(fake_aa_grid, self.resolution, self.delta_s)
        e1 = tf_energy.bond_energy(ref_coords, bond_ndx, self.bond_params)
        e2 = tf_energy.angle_energy(ref_coords, angle_ndx, self.angle_params)
        e3 = tf_energy.dih_energy(ref_coords, dih_ndx, self.dih_params)
        e4 = tf_energy.lj_energy(ref_coords, lj_ndx, self.lj_params)
        return e1,e2,e3,e4

    @tf.function
    def crit_loss_wass(self, critic_real, critic_fake, mask):
        critic_real = critic_real * mask
        critic_fake = critic_fake * mask
        loss_on_generated = tf.reduce_mean(critic_fake)
        loss_on_real = tf.reduce_mean(critic_real)

        loss = loss_on_generated - loss_on_real
        return loss

    @tf.function
    def epsilon_penalty(self, epsilon, critic_real_outputs, mask):
        if epsilon > 0:
            penalties = tf.square(critic_real_outputs) * mask
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
        penalty = tf.reduce_mean(penalty)

        return penalty

    # ## Training steps
    @tf.function
    def iterate_gen(self, acc, elems):
        #Unpack accumulator and elements
        fake_aa_grid, real_aa_grid, cg_features, g_loss_wass = acc
        target_pos, target_type, aa_featvec, repl, mask, bond_ndx, angle_ndx, dih_ndx, lj_ndx = elems

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

        #Gen Loss
        critic_input_fake = tf.concat([fake_atom, fake_c], axis=-1)
        critic_fake = self.critic(critic_input_fake, training=True)
        g_loss_wass = self.gen_loss_wass(critic_fake, mask)

        return fake_aa_grid, real_aa_grid, cg_features, g_loss_wass

    @tf.function
    def train_step_generator(self, elems, initial, energy_ndx, ref_coords):
        #add a term to the initial values to store loss
        initial += (tf.constant(0.0),)
        #with tf.GradientTape() as gen_tape:
            #iterate over sequence (recurrent training)
            #fake_aa_grid, real_aa_grid, _, g_loss_wass = tf.scan(self.iterate_gen, elems, initial)



        e1,e2,e3,e4 = self.energy(ref_coords, energy_ndx)


        #update weights
        #gradients_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        #self.gen_opt.apply_gradients(zip(gradients_generator, self.generator.trainable_variables))
        return e1,e2,e3,e4

    # @tf.function
    def iterate_critic(self, acc, elems):
        #Unpack accumulator and elements
        fake_aa_grid, real_aa_grid, cg_features, c_loss_wass, c_loss_grad, c_loss_eps = acc
        target_pos, target_type, aa_featvec, repl, mask, _, _, _, _ = elems

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
        critic_fake = self.critic(critic_input_fake, training=True)

        critic_input_real = tf.concat([target_atom, real_c], axis=-1)
        critic_real = self.critic(critic_input_real, training=True)

        c_loss_wass = self.crit_loss_wass(critic_real, critic_fake, mask)
        c_loss_grad = self.gradient_penalty(critic_input_real, critic_input_fake, mask)
        c_loss_eps = self.epsilon_penalty(1e-3, critic_real, mask)

        return fake_aa_grid, real_aa_grid, cg_features, c_loss_wass, c_loss_grad, c_loss_eps

    #somehow doesn't work with @tf.function ...
    # @tf.function
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

        #update weights
        gradients_critic = critic_tape.gradient(c_loss, self.critic.trainable_variables)
        self.crit_opt.apply_gradients(zip(gradients_critic, self.critic.trainable_variables))
        return c_loss, c_loss_wass, c_loss_grad, c_loss_eps

    @tf.function
    def predict(self, target_type, atom_pos, atom_featvec, bead_pos, bead_featvec):

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
    def validate(self):
        start = timer()
        print("Saving samples in {}".format(self.samples_dir), "...", end='')

        #u_bm = deepcopy(self.u_val)
        bm_iter = self.u_val.traversal(train=False, mode="init", batch=True)
        for batch in bm_iter:
            atoms, target_type, atom_pos, atom_featvec, bead_pos, bead_featvec = batch
            new_pos = self.predict(target_type, atom_pos, atom_featvec, bead_pos, bead_featvec)
            new_pos = new_pos.numpy()[:,0,:]
            self.u_val.update_pos(atoms, new_pos)

        for u in self.u_val.collection:
            u.write_gro_file(self.samples_dir + "/"+u.name+"_init_" + str(self.checkpoint_gen.step.numpy()) + ".gro")

        bm_iter = self.u_val.traversal(train=False, mode="gibbs", batch=True)
        for batch in bm_iter:
            atoms, target_type, atom_pos, atom_featvec, bead_pos, bead_featvec = batch
            new_pos = self.predict(target_type, atom_pos, atom_featvec, bead_pos, bead_featvec)
            new_pos = new_pos.numpy()[:,0,:]
            self.u_val.update_pos(atoms, new_pos)

        for u in self.u_val.collection:
            u.write_gro_file(self.samples_dir + "/"+u.name+"_gibbs_" + str(self.checkpoint_gen.step.numpy()) + ".gro")

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
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
class GAN():

    def __init__(self, name, model_type, u_train, u_val, z_dim, bs, sigma, resolution, delta_s, rand_rot):

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.compat.v1.Session(config=config)

        self.name = name

        self.z_dim = z_dim
        self.sigma = sigma
        self.resolution = resolution
        self.delta_s = delta_s

        self.rand_rot = rand_rot


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

        #data_generator = lambda: self.u_train.generator(train=True, mode="init", rand_rot=rand_rot)
        data_generator = lambda: self.u_train.generator_combined(train=True, rand_rot=rand_rot)

        ds_train = tf.data.Dataset.from_generator(data_generator, output_types=(tf.float32,   # target position
                                                                              tf.float32,   # target type
                                                                              tf.float32,   # env atoms positions
                                                                              tf.float32,   # env atoms features
                                                                              tf.float32,   # env beads positions
                                                                              tf.float32,   # env bead features
                                                                                tf.bool,  # replace vector
                                                                                tf.int32,  # bond indices
                                                                                tf.int32,  # angle indices
                                                                                tf.int32,  # dih indices
                                                                                tf.int32))  # lj indices


        ds_train = ds_train.batch(bs, drop_remainder=True)
        ds_train = ds_train.map(lambda target_pos, target_type, atom_pos, atom_featvec, bead_pos, bead_featvec, repl, b_ndx, a_ndx, d_ndx, lj_ndx: (
            tf_utils.prepare_target(target_pos, target_type, self.sigma, self.resolution, self.delta_s),
            target_type,
            tf_utils.prepare_grid(atom_pos, atom_featvec, bead_pos, bead_featvec, self.sigma, self.resolution, self.delta_s),
            repl,
            atom_pos,
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

    def train(self, energy_prior, n_start_prior, n_fade_in_prior, n_steps, n_steps_pre, n_critic, n_tensorboard, n_save, n_val, n_gibbs, bm_mode):

        print("Training for n_steps={}, n_critic={}".format(n_steps, n_critic))
        #copyfile("./config.ini", self.logs_dir + "/config_" + str(self.checkpoint_gen.step.numpy()) + ".ini")
        with open('./' + self.name + '/config.ini', 'a') as f:
            f.write("# started at step: " + str(self.checkpoint_gen.step.numpy()))

        if int(self.checkpoint_gen.step) == 1:
            print("pretraining...")
            for step in range(1, n_steps_pre+1):
                real_atom, target_label, env, _, _, _ = next(self.ds_iter)
                g_loss_super, b_loss, a_loss, d_loss, l_loss = self.train_step_generator_supervised(real_atom, env, target_label)
                c_loss, c_loss_wass, c_loss_grad, c_loss_eps = self.train_step_critic(real_atom, env, target_label)
                print("step ", step, g_loss_super.numpy(), b_loss.numpy(), a_loss.numpy(), d_loss.numpy(), l_loss.numpy(), "C: ", c_loss.numpy(), c_loss_wass.numpy(), c_loss_grad.numpy(), c_loss_eps.numpy())


        prior_weight = tf.Variable(0.0)

        for step in range(1, n_steps+1):

            # train critic
            start = timer()
            for i_critic in range(n_critic):
                real_atom, target_label, env, _, _, _ = next(self.ds_iter)
                c_loss, c_loss_wass, c_loss_grad, c_loss_eps = self.train_step_critic(real_atom, env, target_label)

                for (_, metric), loss in zip(self.c_metrics, [c_loss, c_loss_wass, c_loss_grad, c_loss_eps]):
                    metric(loss)


            # train generator
            if energy_prior and int(self.checkpoint_gen.step) >= n_start_prior:
                if int(self.checkpoint_gen.step) < n_start_prior + n_fade_in_prior:
                    prior_weight.assign(0.01 * (int(self.checkpoint_gen.step) - n_start_prior)/n_fade_in_prior)
                else:
                    prior_weight.assign(0.01)
            else:
                prior_weight.assign(0.0)
            real_atom, target_label, env, repl, real_coords, energy_ndx = next(self.ds_iter)
            _, g_loss_tot, g_loss_w, b_loss, a_loss, d_loss, l_loss = self.train_step_generator(env, target_label, repl, real_coords, energy_ndx, prior_weight)
            for (_, metric), loss in zip(self.g_metrics, [g_loss_tot, g_loss_w, b_loss, a_loss, d_loss, l_loss]):
                metric(loss)

            end = timer()

            print(int(self.checkpoint_gen.step), "D: ", c_loss.numpy(), c_loss_wass.numpy(), c_loss_grad.numpy(), c_loss_eps.numpy(), "G: ",
                  g_loss_tot.numpy(), g_loss_w.numpy(), b_loss.numpy(), a_loss.numpy(), d_loss.numpy(), l_loss.numpy(), "time:", end - start)

            self.loss_dict["critic_tot"].append(c_loss.numpy())
            self.loss_dict["critic_wass"].append(c_loss_wass.numpy())
            self.loss_dict["critic_gp"].append(c_loss_grad.numpy())
            self.loss_dict["critic_eps"].append(c_loss_eps.numpy())
            self.loss_dict["gen_tot"].append(g_loss_tot.numpy())
            self.loss_dict["gen_wass"].append(g_loss_w.numpy())
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
                self.validate(n_gibbs, bm_mode=bm_mode)

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
    def gen_loss_wass(self, critic_fake):
        return tf.reduce_mean(-1. * critic_fake)

    @tf.function
    def gen_supervised_loss(self, real_frames, fake_frames):
        fine_frames_normed = tf.divide(real_frames, tf.reduce_sum(real_frames, axis=[1, 2, 3], keepdims=True))
        fake_frames_normed = tf.divide(fake_frames, tf.reduce_sum(fake_frames, axis=[1, 2, 3], keepdims=True))
        g_loss_super = tf.reduce_mean(tf.keras.losses.KLD(fine_frames_normed, fake_frames_normed))
        return g_loss_super

    @tf.function
    def gen_loss_energy(self, real_coords, fake_coords, energy_ndx):
        bond_ndx, angle_ndx, dih_ndx, lj_ndx = energy_ndx

        #tf.print(real_coords)
        #tf.print(fake_coords)
        #print(real_coords)
        #print(fake_coords)

        e_bond_real = tf_energy.bond_energy(real_coords, bond_ndx, self.bond_params)
        e_angle_real = tf_energy.angle_energy(real_coords, angle_ndx, self.angle_params)
        e_dih_real = tf_energy.dih_energy(real_coords, dih_ndx, self.dih_params)
        e_lj_real = tf_energy.lj_energy(real_coords, lj_ndx, self.lj_params)

        e_bond_fake = tf_energy.bond_energy(fake_coords, bond_ndx, self.bond_params)
        e_angle_fake = tf_energy.angle_energy(fake_coords, angle_ndx, self.angle_params)
        e_dih_fake = tf_energy.dih_energy(fake_coords, dih_ndx, self.dih_params)
        e_lj_fake = tf_energy.lj_energy(fake_coords, lj_ndx, self.lj_params)

        bond_loss = tf.abs(e_bond_real - e_bond_fake)
        angle_loss = tf.abs(e_angle_real - e_angle_fake)
        dih_loss = tf.abs(e_dih_real - e_dih_fake)
        lj_loss = tf.abs(e_lj_real - e_lj_fake)

        #return e_bond_real, e_angle_real, e_dih_real, e_lj_real
        return bond_loss, angle_loss, dih_loss, lj_loss


    @tf.function
    def crit_loss_wass(self, critic_real, critic_fake):
        loss_on_generated = tf.reduce_mean(critic_fake)
        loss_on_real = tf.reduce_mean(critic_real)

        loss = loss_on_generated - loss_on_real
        return loss

    @tf.function
    def epsilon_penalty(self, epsilon, critic_real_outputs):
        if epsilon > 0:
            penalties = tf.square(critic_real_outputs)
            penalty = epsilon * tf.reduce_mean(penalties)
            return penalty
        return 0

    @tf.function
    def gradient_penalty(self, input_real, input_fake, target=1.0, use_wgan_lp_loss=False):
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

        penalty = tf.reduce_mean(penalties)

        return penalty

    @tf.function
    def train_step_generator_supervised(self, real_atom, c, l):
        z = tf.random.normal([tf.shape(c)[0], self.z_dim])

        with tf.GradientTape() as gen_tape:
            fake_atom = self.generator([z, c, l])
            fake_atom = tf.reduce_sum(fake_atom, axis=-1, keepdims=True)
            real_atom = tf.reduce_sum(real_atom, axis=-1, keepdims=True)

            g_loss_super = self.gen_supervised_loss(real_atom, fake_atom)

        gradients_generator = gen_tape.gradient(g_loss_super, self.generator.trainable_variables)
        #gradients_generator = [tf.clip_by_value(grad, -1E8, 1E8) for grad in gradients_generator]
        self.gen_opt.apply_gradients(zip(gradients_generator, self.generator.trainable_variables))
        return g_loss_super

    @tf.function
    def train_step_generator(self, c, l, repl, real_coords, energy_ndx, prior_weight):
        z = tf.random.normal([tf.shape(c)[0], self.z_dim])

        with tf.GradientTape() as gen_tape:
            fake_atom = self.generator([z, c, l], training=True)
            fake_coord = tf_utils.average_blob_pos(tf.reduce_sum(fake_atom, axis=-1, keepdims=True), self.resolution, self.delta_s)
            fake_coords = tf.where(repl, real_coords, fake_coord)


            critic_input = tf.concat([fake_atom, c], axis=-1)
            critic_fake = self.critic(critic_input)

            g_loss_wass = self.gen_loss_wass(critic_fake)

            bond_loss, angle_loss, dih_loss, lj_loss = self.gen_loss_energy(real_coords, fake_coords, energy_ndx)

            g_loss_wass = tf.reduce_sum(g_loss_wass)
            bond_loss = tf.reduce_sum(bond_loss)
            angle_loss = tf.reduce_sum(angle_loss)
            dih_loss = tf.reduce_sum(dih_loss)
            lj_loss = tf.reduce_sum(lj_loss)

            gen_loss = g_loss_wass + prior_weight*(bond_loss + angle_loss + dih_loss + lj_loss)

        gradients_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        #gradients_generator = [tf.clip_by_value(grad, -1E8, 1E8) for grad in gradients_generator]
        self.gen_opt.apply_gradients(zip(gradients_generator, self.generator.trainable_variables))
        return fake_atom, gen_loss, g_loss_wass, bond_loss, angle_loss, dih_loss, lj_loss

    @tf.function
    def train_step_critic(self, real_atom, c, l):
        z = tf.random.normal([tf.shape(c)[0], self.z_dim])

        with tf.GradientTape() as critic_tape:
            fake_atom = self.generator([z, c, l])

            critic_input_fake = tf.concat([fake_atom, c], axis=-1)
            critic_fake = self.critic(critic_input_fake, training=True)

            critic_input_real = tf.concat([real_atom, c], axis=-1)
            critic_real = self.critic(critic_input_real, training=True)

            c_loss_wass = self.crit_loss_wass(critic_real, critic_fake)
            c_loss_grad = self.gradient_penalty(critic_input_real, critic_input_fake)
            c_loss_eps = self.epsilon_penalty(1e-3, critic_real)

            c_loss = c_loss_wass + 10. * c_loss_grad + c_loss_eps

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
    def validate(self, n_gibbs, bm_mode="normal"):
        start = timer()
        print("Saving samples in {}".format(self.samples_dir), "...", end='')

        #u_bm = deepcopy(self.u_val)
        bm_iter = self.u_val.make_batch(mode="init")
        for batch in bm_iter:
            atoms, target_type, atom_pos, atom_featvec, bead_pos, bead_featvec = batch
            if self.rand_rot:
                rot_mat = self.u_val.samples[0].rand_rot_mat()
                atom_pos = [np.dot(pos, rot_mat) for pos in atom_pos]
                bead_pos = [np.dot(pos, rot_mat) for pos in bead_pos]
            new_pos = self.predict(target_type, atom_pos, atom_featvec, bead_pos, bead_featvec)
            new_pos = new_pos.numpy()[:,0,:]
            if self.rand_rot:
                new_pos = np.dot(new_pos, rot_mat.T)
            self.u_val.update_pos(atoms, new_pos, bm_mode="normal", temp=568)

        for u in self.u_val.samples:
            u.write_gro_file(self.samples_dir + "/"+u.name+"_init_" + str(self.checkpoint_gen.step.numpy()) + ".gro")

        for n in range(0, n_gibbs):
            bm_iter = self.u_val.make_batch(mode="gibbs")
            for batch in bm_iter:
                atoms, target_type, atom_pos, atom_featvec, bead_pos, bead_featvec = batch
                if self.rand_rot:
                    rot_mat = self.u_val.samples[0].rand_rot_mat()
                    atom_pos = [np.dot(pos, rot_mat) for pos in atom_pos]
                    bead_pos = [np.dot(pos, rot_mat) for pos in bead_pos]
                new_pos = self.predict(target_type, atom_pos, atom_featvec, bead_pos, bead_featvec)
                new_pos = new_pos.numpy()[:,0,:]
                if self.rand_rot:
                    new_pos = np.dot(new_pos, rot_mat.T)
                self.u_val.update_pos(atoms, new_pos, bm_mode=bm_mode, temp=568)

            for u in self.u_val.samples:
                u.write_gro_file(self.samples_dir + "/"+u.name+"_gibbs"+str(n)+"_" + str(self.checkpoint_gen.step.numpy()) + ".gro")

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
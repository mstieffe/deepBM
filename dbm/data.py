import numpy as np
import os
import pickle
import mdtraj as md
import networkx as nx
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from dbm.ff import *
from dbm.universe1 import *
from copy import deepcopy
from operator import add
from pathlib import Path


class Data():

    def __init__(self, cfg, save=True):

        start = timer()

        self.cfg = cfg
        self.aug = int(cfg.getboolean('universe', 'aug'))
        self.align = int(cfg.getboolean('universe', 'align'))
        self.order = cfg.get('universe', 'order')
        self.cutoff = cfg.getfloat('universe', 'cutoff')
        self.kick = cfg.getfloat('universe', 'kick')

        #forcefield
        self.ff_name = cfg.get('forcefield', 'ff_file')
        self.ff_path = Path("./forcefield") / self.ff_name
        self.ff = FF(self.ff_path)

        self.desc = '_aug={}_align={}_order={}_cutoff={}_kick={}_ff={}.pkl'.format(self.aug,
                                                                              self.align,
                                                                              self.order,
                                                                              self.cutoff,
                                                                              self.kick,
                                                                              self.ff_name)

        #samples
        self.dirs_train = [Path("./data/") / d.replace(" ", "") for d in cfg.get('data', 'train_data').split(",")]
        self.dirs_val = [Path("./data/") / d.replace(" ", "") for d in cfg.get('data', 'val_data').split(",")]
        self.dir_processed = Path("./data/processed")
        self.dir_processed.mkdir(exist_ok=True)

        self.samples_train, self.samples_val = [], []
        self.dict_train, self.dict_val = {}, {}
        for path in self.dirs_train:
            self.dict_train[path.stem] = self.get_samples(path, save=save)
        self.samples_train = list(itertools.chain.from_iterable(self.dict_train.values()))
        for path in self.dirs_val:
            self.dict_val[path.stem] = self.get_samples(path, save=save)
        self.samples_val = list(itertools.chain.from_iterable(self.dict_val.values()))

        #find maximums for padding
        self.max = self.get_max_dict()

        print("Successfully created universe! This took ", timer()-start, "secs")

    def get_samples(self, path, save=False):
        name = path.stem + self.desc
        processed_path = self.dir_processed / name

        if processed_path.exists():
            with open(processed_path, 'rb') as input:
                samples = pickle.load(input)
            print("Loaded train universe from " + str(processed_path))
        else:
            samples = []
            cg_dir = path / "cg"
            aa_dir = path / "aa"
            for cg_path in cg_dir.glob('*.gro'):
                aa_path = aa_dir / cg_path.name
                path_dict = {'dir': path, 'cg_path': cg_path, 'file_name': cg_path.name}
                if aa_path.exists():
                    path_dict['aa_path'] = aa_path
                else:
                    path_dict['aa_path'] = None
                u = Universe(self.cfg, path_dict, self.ff)
                samples.append(u)
            if save:
                with open(processed_path, 'wb') as output:
                    pickle.dump(samples, output, pickle.HIGHEST_PROTOCOL)
        return samples

    def get_max_dict(self):
        keys = ['seq_len',
                'beads_loc_env',
                'atoms_loc_env',
                'bonds_per_atom',
                'angles_per_atom',
                'dihs_per_atom',
                'ljs_per_atom',
                'bonds_per_bead',
                'angles_per_bead',
                'dihs_per_bead',
                'ljs_per_bead']
        max_dict = dict([(key, 0) for key in keys])

        samples = self.samples_train + self.samples_val

        for sample in samples:
            for bead in sample.beads:
                max_dict['seq_len'] = max(len(sample.aa_seq_heavy[bead]), len(sample.aa_seq_hydrogens[bead]), max_dict['seq_len'])
                max_dict['beads_loc_env'] = max(len(sample.loc_envs[bead].beads), max_dict['beads_loc_env'])
                max_dict['atoms_loc_env'] = max(len(sample.loc_envs[bead].atoms), max_dict['atoms_loc_env'])

                for aa_seq in [sample.aa_seq_heavy[bead], sample.aa_seq_hydrogens[bead]]:
                    bonds_ndx, angles_ndx, dihs_ndx, ljs_ndx = [], [], [], []
                    for atom in aa_seq:
                        f = sample.aa_features[atom]
                        max_dict['bonds_per_atom'] = max(len(f.energy_ndx_gibbs['bonds']), max_dict['bonds_per_atom'])
                        max_dict['angles_per_atom'] = max(len(f.energy_ndx_gibbs['angles']), max_dict['angles_per_atom'])
                        max_dict['dihs_per_atom'] = max(len(f.energy_ndx_gibbs['dihs']), max_dict['dihs_per_atom'])
                        max_dict['ljs_per_atom'] = max(len(f.energy_ndx_gibbs['ljs']), max_dict['ljs_per_atom'])
                        bonds_ndx += f.energy_ndx_gibbs['bonds']
                        angles_ndx += f.energy_ndx_gibbs['angles']
                        dihs_ndx += f.energy_ndx_gibbs['dihs']
                        ljs_ndx += f.energy_ndx_gibbs['ljs']
                    max_dict['bonds_per_bead'] = max(len(set(bonds_ndx)), max_dict['bonds_per_bead'])
                    max_dict['angles_per_bead'] = max(len(set(angles_ndx)), max_dict['angles_per_bead'])
                    max_dict['dihs_per_bead'] = max(len(set(dihs_ndx)), max_dict['dihs_per_bead'])
                    max_dict['ljs_per_bead'] = max(len(set(ljs_ndx)), max_dict['ljs_per_bead'])

        return max_dict

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
        self.samples_train = list(self.dict_train.values())
        for path in self.dirs_val:
            self.dict_val[path.stem] = self.get_samples(path, save=save)
        self.samples_val = list(self.dict_val.values())

        #find maximums for padding
        self.max = self.get_max_dict()

        print("Successfully created universe! This took ", timer()-start, "secs")

    def get_samples(self, path, save=False):
        name = path.stem + self.desc
        processed_path = self.dir_processed / name

        if processed_path.exists():
            with open(processed_path, 'rb') as input:
                samples = pickle.load(input)
            print("Loaded train universe from " + processed_path)
        else:
            samples = []
            cg_dir = path / "cg"
            aa_dir = path / "aa"
            for cg_path in cg_dir.glob('*.gro'):
                aa_path = aa_dir / cg_path.name
                if aa_path.exists():
                    u = Universe(self.cfg, cg_path, aa_path, self.ff)
                else:
                    u = Universe(self.cfg, cg_path, None, self.ff)
                samples.append(u)
            if save:
                with open(processed_path, 'wb') as output:
                    pickle.dump(samples, output, pickle.HIGHEST_PROTOCOL)
        return samples

    def get_max_dict(self):
        samples = self.samples_train + self.samples_val
        max = {}
        max['seq_len'] = max([u.max_seq_len for u in samples])
        max['atoms'] = max([u.max_atoms for u in samples])
        max['beads'] = max([u.max_beads for u in samples])
        max['bonds_pb'] = max([u.max_bonds_pb for u in samples])
        max['angles_pb'] = max([u.max_angles_pb for u in samples])
        max['dihs_pb'] = max([u.max_dihs_pb for u in samples])
        max['ljs_pb'] = max([u.max_ljs_pb for u in samples])
        max['bonds'] = max([u.max_bonds for u in samples])
        max['angles'] = max([u.max_angles for u in samples])
        max['dihs'] = max([u.max_dihs for u in samples])
        max['ljs'] = max([u.max_ljs for u in samples])
        return max

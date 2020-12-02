import logging
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter


class OutputHandler(object):

    def __init__(self, model_name, keep_n_checkpoints, output_dir=""):
        self._log = logging.getLogger(__name__)
        self.model_name = model_name
        self.output_dir = Path(output_dir) / self.model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        #self.model_dir = self._init_dir(model_name)
        self.checkpoint_dir = self._init_dir("checkpoints")
        self.samples_dir = self._init_dir("samples")
        self.logs_dir = self._init_dir("logs")
        self.logs_writer = SummaryWriter(log_dir=self.logs_dir / 'train')
        self.logs_writer_val = SummaryWriter(log_dir=self.logs_dir / 'val')
        self.logs_writer_test = SummaryWriter(log_dir=self.logs_dir / 'test')
        self.keep_n_checkpoints = keep_n_checkpoints

    def _init_dir(self, directory: str):
        p = self.output_dir / directory
        p.mkdir(parents=True, exist_ok=True)
        return p

    def make_checkpoint(self, step, dictionary):
        outfile = self.checkpoint_dir / f"checkpoint_{step}.ckpt"
        torch.save(dictionary, outfile)
        return outfile

    def prune_checkpoints(self):
        checkpoints = list(self.checkpoint_dir.glob("*.ckpt"))
        steps = []
        for ckpt in checkpoints:
            fname = str(Path(ckpt).name)
            steps.append(int("".join(list(c for c in filter(str.isdigit, fname)))))
        while len(steps) > self.keep_n_checkpoints:
            oldest = min(steps)
            oldest_path = self.checkpoint_dir.joinpath(
                "checkpoint_{}.ckpt".format(oldest)
            )
            self._log.info("Removing oldest checkpoint {}".format(oldest_path))
            oldest_path.unlink()
            steps.remove(oldest)

    def latest_checkpoint(self):
        checkpoints = list(self.checkpoint_dir.glob("*.ckpt"))
        if len(checkpoints) > 0:
            steps = []
            for ckpt in checkpoints:
                fname = str(Path(ckpt).name)
                steps.append(int("".join(list(c for c in filter(str.isdigit, fname)))))
            latest = max(steps)
            latest_ckpt = self.checkpoint_dir / f"checkpoint_{latest}.ckpt"
            return latest_ckpt
        else:
            return None

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None, mode='train'):
        if mode == 'train':
            writer = self.logs_writer
        elif mode == 'val':
            writer = self.logs_writer_val
        elif mode == 'test':
            writer = self.logs_writer_test
        else:
            raise ValueError('unknown mode', mode)
        writer.add_scalar(tag, scalar_value, global_step=global_step, walltime=walltime)

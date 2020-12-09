import matplotlib.pyplot as plt
import numpy as np

class Fig():

    def __init__(self, path, n_plots, figsize=(12,12)):

        self.path = path
        self.fig = plt.figure(figsize=figsize)
        self.n_ax = int(np.ceil(np.sqrt(n_plots)))
        self.current_ax = 1

        self.color_bm = "red"
        self.color_ref = "black"
        self.lw = 2

    def save(self):
        plt.savefig(self.path)
        plt.close()

    def add_plot(self, dstr, dict, ref_dstr=None):
        # adding a plot to the fig

        ax = self.fig.add_subplot(self.n_ax, self.n_ax, self.current_ax)
        ax.set_title(dict["title"], fontsize=12)

        values = list(dstr.values())
        keys = list(dstr.keys())
        ax.plot(keys, values, label="bm", color=self.color_bm, linewidth=self.lw, linestyle='-')

        if ref_dstr:
            ref_values = list(ref_dstr.values())
            ref_keys = list(ref_dstr.keys())
            ax.plot(ref_keys, ref_values, label="ref", color=self.color_ref, linewidth=self.lw, linestyle='-')

        ax.set_xlabel(dict["xlabel"])
        ax.set_ylabel(dict["ylabel"])
        plt.legend()
        plt.tight_layout()

        self.current_ax += 1


    def plot(self, dstr, dict, ref_dstr=None):
        # function for plotting a dstr

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(dict["title"], fontsize=4)

        values = list(dstr.values())
        keys = list(dstr.keys())
        ax.plot(keys, values, label="bm", color="red", linewidth=2, linestyle='-')

        if ref_dstr:
            ref_values = list(ref_dstr.values())
            ref_keys = list(ref_dstr.keys())
            ax.plot(ref_keys, ref_values, label="ref", color="black", linewidth=2, linestyle='-')

        ax.set_xlabel(dict["xlabel"])
        ax.set_ylabel(dict["ylabel"])
        plt.legend()
        plt.tight_layout()
        plt.savefig(dict["name"] + ".pdf")
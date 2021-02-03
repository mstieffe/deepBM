# deepBM

This is the code for the deepBackmap (DBM) algorithm accompanying our recent publications
- [Adversarial reverse mapping of equilibrated condensed-phase molecular structures](https://iopscience.iop.org/article/10.1088/2632-2153/abb6d4/meta) 
- [Adversarial reverse mapping of condensed-phase molecular structures: Chemical transferability](https://arxiv.org/abs/2101.04996) 

DeepBackmap is a deep neural network based approach for backmapping of condened-phase molecular structures. We use generative adversarial networks to learn the Boltzmann distribution from training data and realize reverse mapping by using the coarse-grained (CG) structure as a conditional input. We use a voxel representation to encode spatial relationships and make use of different feature channels typical for convolutional neural networks to encode information of the molecular topology. The loss function of the generator is augmented with a term penalizing configurations with high potential energy. A regular discretization of 3D space prohibits scaling to larger spatial structures. Therefore, we use an autoregressive approach that reconstructs the fine-grained structure incrementally, atom by atom.

## python packages

To create a conda environment with all the required python packages you can use the env.yml file:

```
conda env create -f env.yml
```

## usage

The code is organized as followes:

### data

The user needs to provide the following data:
- snapshots of CG molecular structures with file extension `.gro`, formatted as described [here](https://manual.gromacs.org/archive/5.0.4/online/gro.html). The files have to be stored in a directory `my_dir` inside `./data/reference_snapshots/my_dir/cg`. If the user wants to train a new model reference AA structure files have to be provided too. The AA structure files are stored inside `./data/reference_snapshots/my_dir/aa` and have be named identical to their corresponding CG structure file.
- for each residue with name `res_name` included in the snapshot a corresponding toplogy file with extension `.itp` has to be provided for both, the AA toplogy and the CG topology. The formatting of the topologx file is described [here](https://manual.gromacs.org/archive/5.0/online/top.html). The files have to be stored inside `./data/aa_top/res_name.itp` and `./data/cg_top/res_name.itp` respectively.
- for each residue a mapping file `.map` is needed that describes the correspondence between CG and AA structures. The file needs to be stored inside `./data/mapping/res_name.map`. 
- parameters specifying the model (such as model name, resolution, regularizer and many more...) are stored in a `config.ini` (see example below for further details).
- the feature mapping and energy terms are specified in a `.txt` file inside `./forcefield/` (see sample below for further details).

### example

In the folllowing, we will train DBM on liquid-phase structures of cumene and octane and then use it for the backmapping of syndiotactic polystyrene (sPS).


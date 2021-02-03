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

#### data preparation

- Snapshots for cumene, octane and sPS be found `./data/reference_snapshots/`. The coarse-grained model of sPS was developed by Fritz *et al*[1]. It represents a polymer as a linear chain, where each monomer is mapped onto two CG beads of different types, denoted A for the chain backbone and B for the phenyl ring. The center of bead A is the center of mass of the CH2 group and the two neighboring CH groups, which are weighted with half of their masses. Bead B is centered at the center of mass of the phenyl group. Cumene is mapped onto three CG beads: Two beads of type A for the backbone, each containing a methyl group and sharing the CH group connected to the phenyl ring, and one bead of type B for the phenyl ring. Octane is mapped onto four beads of type A, where neighboring A beads share a CH2 group.

- Next, the mapping file is generated. As an example, the mapping for looks like this:
```
[map]
    1  H_AR 2   A
    2  C_AR 2   A
    3  C_AR 2   A
    4  H_AR 2   A
    5  C_AR 2   A
    6  H_AR 2   A
    7  C_AR 2   A
    8  C_AR 2   A
    9  H_AR 2   A
   10  C_AR 2   A
   11  H_AR 2   A
   12    C  1   B
   13    H  1   B
   14    C  1   B
   15    H  1   B
   16    H  1   B
   17    H  1   B
   18    C  3   B
   19    H  3   B
   20    H  3   B
   21    H  3   B
[/map]

[align]
;bead	fixpoint
1       3
2       1
3	1
[/align]

[mult]
;bead	multiples
1       1
2       1
3	1
[/mult]
```




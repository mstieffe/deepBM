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

- We place the toplogy files for all residues used into the corresponding directories. For example, the residue for cumene is named `G065` and we will have one file `G065.itp` inside `./data/aa_top/` as well as `./data/cg_top/`, to define the topologies for the AA as well as the CG structure.

- Next, the mapping file is generated. As an example, the mapping for looks like this:
```
[map]
;atom_index atom_type bead_index bead_type 
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
;bead_index	fixpoint
1       3
2       1
3	1
[/align]

[mult]
;bead_index	multiples
1       1
2       1
3	1
[/mult]
```
While CG force fields might lead to the sharing of an atom between two neighboring
beads, the reconstruction of the atom is assigned to only one of the two beads. The mapping is defined between `[map][/map]`. Additionally, we can define a preference axis for each bead to reduce the rotational degrees of freedom. This preference axis can be defined by the position of the central bead and the difference vector to any other bead, which is specified between `[align][/align]`. Furthermore, we can use data augmentation and can increase the number of occurances of a given bead in the training set by integer multiples defined inside `[mult][/mult]`.

- Then we have to define the feature mapping and place it as a `.txt` file inside the directory `./forcefield`. As an example we could use:
```
[general]
; Name      nrexcl
ff2           2
[/general]

[atom_types]
;name	channel   mass      charge       sigma      epsilon
C	-1       12.0110     0.0000          0.3207      0.3519
C_AR	-1       12.0110    -0.1150          0.3550      0.2940
H	-1        1.0080     0.0000          0.2318      0.3180
H_AR	-1        1.0080     0.1150          0.2420      0.1260
[/atom_types]

[bond_types]
; i     j	channel  func        b0          kb
C      C	0        1       0.153000      1000.00
C      H	1        1       0.110000      1000.00
C_AR   C_AR	2     	 1       0.139000      1000.00
C_AR   H_AR	1     	 1       0.108000      1000.00
C      C_AR	3    	 1       0.151000      1000.00
[/bond_types]

[angle_types]
; i     j      k	channel  func       th0         cth
H      C      H		4       1       109.45       306.40
C      C      H 	5       1       109.45       448.90
C      C      C   	6     	1       111.00       530.30
C_AR   C      H  	7       1       109.45       366.90
C      C      C_AR 	8   	1       109.45       482.30
C      C_AR   C_AR  	9   	1       120.00       376.60
C_AR   C_AR   C_AR  	10  	1       120.00       376.60
C_AR   C_AR   H_AR  	11  	1       120.00       418.80
[/angle_types]


[dihedral_types]
; i    j     k     l	channel func 
C      C     C     C	12      1    0.0000   6.0000   3.0000
H      C     C     C	12      1    0.0000   6.0000   3.0000
C_AR   C_AR  C_AR  C_AR	13      2    0.0000 167.4000
C_AR   C_AR  C_AR  C	13      2    0.0000 167.4000
C_AR   C_AR  C_AR  H_AR	13      2    0.0000 167.4000
[/dihedral_types]

[lj_types]
; i     j       channel
C       C_AR    14
C       H_AR    15
C       C       14
C       H       15
C_AR    H_AR    15
C_AR    C_AR    14
C_AR    H       15
H_AR    H_AR    16
H_AR    H       16
H       H       16
[/lj_types]

[bead_types]
;name	channel
B	17
S	18
[/bead_types]
```



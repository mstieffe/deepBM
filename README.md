# deepBM

This is the code for the deepBackmap algorithm accompanying our recent publications
- [Adversarial reverse mapping of equilibrated condensed-phase molecular structures](https://iopscience.iop.org/article/10.1088/2632-2153/abb6d4/meta) 
- [Adversarial reverse mapping of condensed-phase molecular structures: Chemical transferability](https://arxiv.org/abs/2101.04996) 

DeepBackmap is a deep neural network based approach for backmapping of condened-phase molecular structures. We use generative adversarial networks to learn the Boltzmann distribution from training data and realize reverse mapping by using the coarse-grained structure as a conditional input. We use a voxel representation to encode spatial relationships and make use of different feature channels typical for convolutional neural networks to encode information of the molecular topology. The loss function of the generator is augmented with a term penalizing configurations with high potential energy. A regular discretization of 3D space prohibits scaling to larger spatial structures. Therefore, we use an autoregressive approach that reconstructs the fine-grained structure incrementally, atom by atom.

## python packages

To create a conda environment with all the required python packages you can use the env.yml file:

```
conda env create -f env.yml
```

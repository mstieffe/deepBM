# deepBM

DeepBackmap is a deep neural network based approach for backmapping of condened-phase molecular structures. We use generative adversarial networks to learn the Boltzmann distribution from training data and realize reverse mapping by using the coarse-grained structure as a conditional input. We use a voxel representation to encode spatial relationships and make use of different feature channels typical for convolutional neural networks to encode information of the molecular topology. The loss function of the generator is augmented with a term penalizing configurations with high potential energy. A regular discretization of 3D space prohibits scaling to larger spatial structures. Therefore, we use an autoregressive approach that reconstructs the fine-grained structure incrementally, atom by atom.

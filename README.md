# MLIPOps

MLIPOps is a collection of PyTorch modules for use in creating machine learning interatomic potentials (MLIPs).  It
is written in Python using PyTorch and Triton to do calculations.  It is based around the following design principles.

- Pure Python.  This avoids the compatibility, maintainability, and distribution challenges that come with compiled
  extensions.
- Highly portable.  It works on any computer that PyTorch can run on.
- Fast performance on any hardware that Triton supports.  Currently that includes NVIDIA and AMD GPUs, with others
  likely to be added in the future.
- Clean, simple code.  We rely on `torch.compile` as the first line of optimization, adding Triton kernels only for
  calculations PyTorch cannot do a satisfactory job of optimizing on its own.  This keeps the code base simple and easy
  to understand.

## Installation

To install from PyPI type

```
pip install mlipops
```

Alternatively, if you want to install from source, check out the code and type the following command from the root
directory.

```
pip install .
```

## Documentation

See the [User Guide](https://openmm.github.io/mlipops/dev/userguide.html) and [API documentation](https://openmm.github.io/mlipops/dev/api.html)
for instructions on how to use MLIPOps.

## Features

These are the currently implemented features.  Expect this list to grow with time.

- Neighbor list construction
- Coulomb interactions
  - No cutoff: charges and dipoles, non-periodic systems
  - Reaction field: charges only, periodic and non-periodic systems
  - Ewald summation: charges and dipoles, periodic systems
  - Particle Mesh Ewald: charges and dipoles, periodic systems
- Ziegler-Biersack-Littmark (ZBL) potential
- DFT-D3(BJ) dispersion
- Arbitrary pairwise potentials
- Batch computation
- Periodic boundary conditions with triclinic boxes
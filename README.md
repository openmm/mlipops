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

To install, check out the code and type the following command from the root directory.

```
pip install .
```

PyPI packages will be coming once the code matures a little more.

## Features

These are the currently implemented features.  Many more are planned.  Expect this list to grow quickly.

- Neighbor list construction
- Coulomb interactions with Particle Mesh Ewald
- Arbitrary pairwise potentials
- Periodic boundary conditions with triclinic boxes
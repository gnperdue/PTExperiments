# Top-level Contents

* `check_coverage.sh` -- run `coverage` over test suite (via `run_tests.py`)
  and report unit test coverage.
* `cleanup.sh` -- remove development detritus, pyc files, etc.
* `dset_viz.py` -- visualize star-galaxy data.
* `ptlib/` -- lib dir for PyTorch code modules.
* `requirements.txt` -- packages needed to run everything.
* `run_tests.py` -- run all tests in `tests/`; generally best to avoid running
  this directly and instead run `check_coverage.sh`.
* `run_vanilla_fashion.sh` -- wrapper script for Fashion MNIST classifier.
* `run_vanilla_stargalaxy.sh` -- wrapper script for the star-galaxy classifier.
* `tests/` -- lib dir for `unittest` modules.
* `vanilla_fashion.py` -- script to run a Fashion MNIST classifier.
* `vanilla_stargalaxy.py` -- script to run a star-galaxy classifier.

## Data

Tensors are stored using PyTorch convnetions - [Number, Channel, Width, Height]

### Star-Galaxy

Images courtesy of Joao Caldeira (provenance unknown to me, but likely known to
him). HDF5 data is available here:

HDF5 data is available here:

https://github.com/gnperdue/RandomData/tree/master/StarGalaxy

e.g.

```
wget https://raw.githubusercontent.com/gnperdue/RandomData/master/StarGalaxy/stargalaxy_real_pt_test.hdf5
wget https://raw.githubusercontent.com/gnperdue/RandomData/master/StarGalaxy/stargalaxy_real_pt_train.hdf5
wget https://raw.githubusercontent.com/gnperdue/RandomData/master/StarGalaxy/stargalaxy_real_pt_valid.hdf5
```

**NOTE** -- the images stored in that Git repo are `uint8`, but the code expects
floating points. There is a script in the `RandomData/StarGalaxy` package for
doing the conversion using the [hdf5_manipulator](https://github.com/gnperdue/hdf5_manipulator)
package of utilities. (Not storing the floats in GitHub because those files are
too large.)

### Fashion MNIST

HDF5 data is available here:

https://github.com/gnperdue/RandomData/tree/master/hdf5

e.g.

```
wget https://raw.githubusercontent.com/gnperdue/RandomData/master/hdf5/fashion_test.hdf5
wget https://raw.githubusercontent.com/gnperdue/RandomData/master/hdf5/fashion_train.hdf5
```

(Use 'test' for 'validation' -- no real 'test' file offered for this dset
because it is _fashion MNIST_ for crying out loud...)

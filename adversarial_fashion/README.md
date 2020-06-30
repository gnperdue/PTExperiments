# Top-level Contents

* `check_coverage.sh` -- run `coverage` over test suite (via `run_tests.py`)
  and report unit test coverage.
* `cleanup.sh` -- remove development detritus, pyc files, ectc.
* `ptlib/` -- lib dir for PyTorch code modules.
* `requirements.txt` -- packages needed to run everything.
* `run_tests.py` -- run all tests in `tests/`; generally best to avoid running
  this directly and instead run `check_coverage.sh`.
* `run_vanilla_fashion.sh` -- wrapper script for Fashion MNIST classifier.
* `tests/` -- lib dir for `unittest` modules.
* `vanilla_fashion.py` -- script to run a Fashion MNIST classifier.

Fashion MNIST HDF5 data is available here:

https://github.com/gnperdue/RandomData/tree/master/hdf5

e.g.

```
wget https://raw.githubusercontent.com/gnperdue/RandomData/master/hdf5/fashion_test.hdf5
wget https://raw.githubusercontent.com/gnperdue/RandomData/master/hdf5/fashion_train.hdf5
```

(Use 'test' for 'validation' -- no real 'test' file offered for this dset
because it is _fashion MNIST_ for crying out loud...)

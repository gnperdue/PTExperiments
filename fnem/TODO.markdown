# TODO

* Modify the policies so they produce a setting value directly. For historical
training, this is sufficient to compare to the datasource target setting. For
live training, we need to convert this to a heat in a way that can be given
to the loss function. The way to do this is to construct a paired "reward"
that is the heat. The loss is the setting squared times the heat to the fourth,
or something like that... at issue is the setting can be negative, and may
need to be "large" (close to 10)... by driving the setting to zero, the loss
could be made small w/o making the heat small. Maybe the loss should be
something like `(setting * heat^2)^2 + heat^4`?
* Need to tie sequence and heat buffer lengths to NN output size - no, this
is not right. Output size is set by the number of actions. Sequence length
impacts network inputs.
* Implement `train` in `SimpleMLP`.

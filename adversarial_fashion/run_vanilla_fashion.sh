#!/bin/bash

DATADIR="${HOME}/Dropbox/Data/RandomData/hdf5"
EXE="vanilla_fashion.py"

ARGS="--batch-size 16"
ARGS+=" --ckpt-path ckpt.tar"
ARGS+=" --data-dir ${DATADIR}"
ARGS+=" --log-level INFO"
ARGS+=" --num-epochs 1"
ARGS+=" --show-progress"

# show exe and args
cat << EOF
python $EXE $ARGS
EOF

python $EXE $ARGS

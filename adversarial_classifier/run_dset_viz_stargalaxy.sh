#!/bin/bash

DATADIR="/Users/perdue/Dropbox/Data/Workspace"

EXE="dset_viz_stargalaxy.py"
ARGS="--batch-size 5"
ARGS+=" --data-dir ${DATADIR}"
# ARGS+=" --file-override ./fgsm.hdf5"
ARGS+=" --num-batches 8"

# show exe and args
cat << EOF
python $EXE $ARGS
EOF

python $EXE $ARGS

echo -e "\a"

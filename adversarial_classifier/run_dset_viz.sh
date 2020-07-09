#!/bin/bash

EXE="dset_viz.py"
ARGS="--batch-size 6"
ARGS+=" --data-dir /Users/perdue/Dropbox/Data/RandomData/StarGalaxy"
ARGS+=" --num-batches 3"

# show exe and args
cat << EOF
python $EXE $ARGS
EOF

python $EXE $ARGS

echo -e "\a"

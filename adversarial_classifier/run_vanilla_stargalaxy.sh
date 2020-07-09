#!/bin/bash

DATADIR="/Users/perdue/Dropbox/Data/StarGalaxy"
EXE="vanilla_stargalaxy.py"
DAT=`date +%s`

ARGS="--batch-size 32"
ARGS+=" --ckpt-path sg_ckpt.tar"
ARGS+=" --data-dir ${DATADIR}"
ARGS+=" --log-freq 5"
ARGS+=" --log-level INFO"
ARGS+=" --num-epochs 1"
ARGS+=" --short-test"
ARGS+=" --show-progress"
ARGS+=" --tnsrbrd-out-dir /tmp/stargalaxy/tnsrbrd${DAT}"
# ARGS+=" --tnsrbrd-out-dir /tmp/fashion/tnsrbrd_test"

# show exe and args
cat << EOF
python $EXE $ARGS
EOF

python $EXE $ARGS

echo -e "\a"

#!/bin/bash

DATADIR="/Users/perdue/Dropbox/Data/RandomData/StarGalaxy"
EXE="attack_stargalaxy.py"
DAT=`date +%s`

ARGS="--ckpt-path sg_ckpt.tar"
ARGS+=" --data-dir ${DATADIR}"
ARGS+=" --log-freq 5"
ARGS+=" --log-level DEBUG"
ARGS+=" --short-test"

# show exe and args
cat << EOF
python $EXE $ARGS
EOF

python $EXE $ARGS

echo -e "\a"

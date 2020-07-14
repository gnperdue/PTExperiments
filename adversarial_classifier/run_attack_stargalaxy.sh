#!/bin/bash

DATADIR="/Users/perdue/Dropbox/Data/RandomData/StarGalaxy"
EXE="attack_stargalaxy.py"
DAT=`date +%s`

ARGS="--ckpt-path sg_ckpt.tar"
ARGS+=" --data-dir ${DATADIR}"
ARGS+=" --git-hash `git describe --abbrev=12 --dirty --always`"
ARGS+=" --log-freq 5"
ARGS+=" --log-level DEBUG"
# ARGS+=" --log-level INFO"
ARGS+=" --short-test"

# show exe and args
cat << EOF
python $EXE $ARGS
EOF

python $EXE $ARGS

echo -e "\a"

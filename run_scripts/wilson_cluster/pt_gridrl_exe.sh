#!/bin/bash

echo "started "`date`" "`date +%s`""

BASEPATH="/home/perdue/PTExperiments/gridworld"

mkdir games
mkdir gridrl
cp ${BASEPATH}/games/*.py ./games/
cp ${BASEPATH}/gridrl/*.py ./gridrl/
cp ${BASEPATH}/nnqrt.py .

EXE="nnqrt.py"
NUM_EPOCHS=1000

ARGS="--num-epochs ${NUM_EPOCHS}"
#ARGS+=" --batch-size ${BATCH_SIZE}"
ARGS+=" --conv"


SNGLRTY="/data/perdue/singularity/gnperdue-singularity_imgs-master-py3_trch041.simg"

cat << EOF
singularity exec --nv $SNGLRTY python3 $EXE $ARGS
EOF
singularity exec --nv $SNGLRTY python3 $EXE $ARGS

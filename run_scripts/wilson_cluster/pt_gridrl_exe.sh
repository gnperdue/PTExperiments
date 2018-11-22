#!/bin/bash

echo "started "`date`" "`date +%s`""

BASEPATH="/home/perdue/PTExperiments/gridworld"

mkdir games
mkdir gridrl
cp ${BASEPATH}/games/*.py ./games/
cp ${BASEPATH}/gridrl/*.py ./gridrl/
cp ${BASEPATH}/nnqrt.py .

EXE="nnqrt.py"
NUM_EPOCHS=100

MODEL_DIR="/data/perdue/minerva/tensorflow/models/test"
DATA_DIR="/data/perdue/minerva/hdf5/201804"
TRAIN_FILE="${DATA_DIR}/hadmultkineimgs_127x94_me1Nmc.hdf5"
EVAL_FILE="${DATA_DIR}/hadmultkineimgs_127x94_me1Omc.hdf5"

ARGS="--num-epochs ${NUM_EPOCHS}"
#ARGS+=" --batch-size ${BATCH_SIZE}"


SNGLRTY="/data/perdue/singularity/gnperdue-singularity_imgs-master-py3_trch041.simg"

cat << EOF
singularity exec --nv $SNGLRTY python3 $EXE $ARGS
EOF
singularity exec --nv $SNGLRTY python3 $EXE $ARGS

#!/bin/bash
EXE="master_run.py"

CKPT="ckpt.tar"
# MODE="TRAIN-LIVE"
# NUMSTEPS=1000
MODE="TRAIN-HISTORICAL"
NUMEPOCHS=1
HISTORYFILE="./reference_files/log_machinewithrule_1545976343.csv.gz"
POLICY="SimpleRuleBased"
SEQSIZE=1

ARGS="--ckpt-path $CKPT"
ARGS+=" --mode $MODE"
# ARGS+=" --num-steps $NUMSTEPS"
ARGS+=" --num-epochs $NUMEPOCHS"
ARGS+=" --data-source-path $HISTORYFILE"
ARGS+=" --policy $POLICY"
ARGS+=" --sequence-size $SEQSIZE"
ARGS+=" --make-plot"

cat << EOF
python $EXE $ARGS
EOF

python $EXE $ARGS

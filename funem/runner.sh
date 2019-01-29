#!/bin/bash

NUMGO=1000
if [[ $# == 1 ]]; then
  NUMGO=$1
fi

EXE="master_run.py"

CKPT="ckpt.tar"
ARGS="--ckpt-path $CKPT"

MODE="TRAIN-LIVE"
NUMSTEPS=$NUMGO
ARGS+=" --num-steps $NUMSTEPS"

# MODE="TRAIN-HISTORICAL"
# NUMEPOCHS=1
# HISTORYFILE="./reference_files/log_machinewithrule_1545976343.csv.gz"
# ARGS+=" --num-epochs $NUMEPOCHS"
# ARGS+=" --data-source-path $HISTORYFILE"

# POLICY="SimpleRuleBased"

POLICY="SimpleMLP"

ARGS+=" --learner $POLICY"
ARGS+=" --mode $MODE"
ARGS+=" --make-plot"
ARGS+=" --show-progress"

cat << EOF
python $EXE $ARGS
EOF

python $EXE $ARGS

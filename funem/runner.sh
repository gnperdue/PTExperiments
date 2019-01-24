#!/bin/bash
EXE="master_run.py"

CKPT="ckpt.tar"
ARGS="--ckpt-path $CKPT"

MODE="TRAIN-LIVE"
NUMSTEPS=1000
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

# python $EXE $ARGS

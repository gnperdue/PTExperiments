#!/bin/bash
EXE="master_run.py"

CKPT="ckpt.tar"
ARGS="--ckpt-path $CKPT"

# MODE="TRAIN-LIVE"
# NUMSTEPS=1000
# ARGS+=" --num-steps $NUMSTEPS"

MODE="TRAIN-HISTORICAL"
NUMEPOCHS=1
HISTORYFILE="./reference_files/log_machinewithrule_1545976343.csv.gz"
ARGS+=" --num-epochs $NUMEPOCHS"
ARGS+=" --data-source-path $HISTORYFILE"

POLICY="SimpleRuleBased"
SEQSIZE=1
ARGS+=" --policy $POLICY"
ARGS+=" --sequence-size $SEQSIZE"

# POLICY="SimpleMLP"
# SEQSIZE=20
# ARGS+=" --policy $POLICY"
# ARGS+=" --sequence-size $SEQSIZE"


ARGS+=" --mode $MODE"
ARGS+=" --make-plot"

cat << EOF
python $EXE $ARGS
EOF

python $EXE $ARGS

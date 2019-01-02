#!/bin/bash
EXE="master_run.py"

CKPT="ckpt.tar"
MODE="TRAIN-LIVE"
NUMSTEPS=1000
POLICY="SimpleRuleBased"
SEQSIZE=1

ARGS="--ckpt-path $CKPT"
ARGS+=" --mode $MODE"
ARGS+=" --num-steps $NUMSTEPS"
ARGS+=" --policy $POLICY"
ARGS+=" --sequence-size $SEQSIZE"
# ARGS+=" --make-plot"

cat << EOF
python $EXE $ARGS
EOF

python $EXE $ARGS

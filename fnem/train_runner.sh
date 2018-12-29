#!/bin/bash
EXE="master_run.py"

CKPT="ckpt.tar"
MODE="TRAIN-LIVE"
NUMSTEPS=1000
POLICY="SimpleRuleBased"

ARGS="--ckpt-path $CKPT"
ARGS+=" --make-plot"
ARGS+=" --mode $MODE"
ARGS+=" --num-steps $NUMSTEPS"
ARGS+=" --policy $POLICY"

cat << EOF
python $EXE $ARGS
EOF

python $EXE $ARGS

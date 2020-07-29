#!/bin/bash

EXE="nnqrt.py"
ARGS="--batch-size 10"
ARGS+=" --buffer 20"
ARGS+=" --ckpt-path ckpt.tar"
ARGS+=" --conv"
# ARGS+=" --epsilon 1.0"
# ARGS+=" --gamma 0.95"
ARGS+=" --game-mode random"
ARGS+=" --game-size 4"
ARGS+=" --learning-rate 0.001"
ARGS+=" --log-level INFO"
ARGS+=" --make-plot"
ARGS+=" --num-epochs 1000"
ARGS+=" --saved-losses-path losses.npy"
ARGS+=" --saved-winpct-path winpct.npy"
ARGS+=" --show-progress"
ARGS+=" --target-network-update 10"

# show exe and args
cat << EOF
python $EXE $ARGS
EOF

python $EXE $ARGS

echo -e "\a"

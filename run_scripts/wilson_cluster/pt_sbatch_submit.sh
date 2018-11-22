#!/bin/bash

SCRIPTKEY=`date +%s`
mkdir job${SCRIPTKEY}

NGPU=1
NODES=gpu2
NODES=gpu3
NODES=gpu4

cat << EOF
sbatch --gres=gpu:${NGPU} --nodelist=${NODES} -A minervag -p gpu pt_gridrl_exe.sh
EOF

# do the thing, etc.
cp pt_gridrl_exe.sh job${SCRIPTKEY}
pushd job${SCRIPTKEY}
sbatch --gres=gpu:${NGPU} --nodelist=${NODES} -A minervag -p gpu pt_gridrl_exe.sh
popd

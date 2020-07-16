#!/bin/bash

DAT=`date +%s`

# clean up synth data from testing
rm -fv fash_synth_*.npy
rm -fv fash_synth_*.h5
rm -fv sg_synth_*.npy
rm -fv sg_synth_*.h5

# cleanup all logs -- could choose to save them eventually...
rm -fv log_vanilla*.txt
rm -fv log_attack*.txt
rm -fv log_run_tests*.txt

# archive training artifacts
mkdir -p archive
FILELIST="sg_ckpt.tar short_test.tar"
for file in $FILELIST
do
  if [[ -e $file ]]; then
    mv -v $file archive/t${DAT}_${file}
  fi
done

# clean up compiled python
DIRS="ptlib tests"
for dir in $DIRS
do
    rm -rfv ${dir}/__pycache__
done

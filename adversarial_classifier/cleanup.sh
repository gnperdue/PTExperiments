#!/bin/bash

DAT=`date +%s`

# clean up synth data from testing
rm -fv synth_mean.npy
rm -fv synth_std.npy
rm -fv synth_test.h5
rm -fv synth_train.h5
rm -fv synth_valid.h5

# cleanup all logs -- could choose to save them eventually...
rm -fv log_vanilla*.txt
rm -fv log_attack*.txt

# archive training artifacts
mkdir -p archive
FILELIST="ckpt.tar short_test.tar"
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

#!/bin/bash
DAT=`date +%s`
mkdir -p archive
rm -f log_nnqrt*.txt
rm -f deepq_targrep_*pdf
rm -f test_losses.npy
rm -f test_winpct.npy

FILELIST="ckpt.tar losses.npy winpct.npy"
for file in $FILELIST
do
  if [[ -e $file ]]; then
    mv $file archive/t${DAT}_${file}
  fi
done

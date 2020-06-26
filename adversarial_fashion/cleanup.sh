#!/bin/bash

DAT=`date +%s`

# cleanup all logs -- could choose to save them eventually...
rm -f log_vanilla*.txt

# archive training artifacts
mkdir -p archive
FILELIST="ckpt.tar"
for file in $FILELIST
do
  if [[ -e $file ]]; then
    mv $file archive/t${DAT}_${file}
  fi
done

# clean up compiled python
DIRS="ptlib tests"
for dir in $DIRS
do
    rm -rf ${dir}/__pycache__
done

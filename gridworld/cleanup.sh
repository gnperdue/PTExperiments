#!/bin/bash
DAT=`date +%s`
mkdir -p archive
mv -f ckpt.tar archive/t${DAT}_ckpt.tar
mv -f losses.npy archive/t${DAT}_losses.npy
mv -f winpct.npy archive/t${DAT}_winpct.npy
rm -f log_nnqrt*.txt
rm -f deepq_targrep_*pdf

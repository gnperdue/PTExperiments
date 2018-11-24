#!/bin/bash
DAT=`date +%s`
mkdir -p archive
rm -f log_nnqrt*.txt
rm -f deepq_targrep_*pdf
rm -f test_losses.npy
rm -f test_winpct.npy
mv ckpt.tar archive/t${DAT}_ckpt.tar
mv losses.npy archive/t${DAT}_losses.npy
mv winpct.npy archive/t${DAT}_winpct.npy

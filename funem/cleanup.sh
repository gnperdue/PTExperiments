#!/bin/bash
rm -fv log_engine_*.csv*
rm -fv tmplog.csv*
rm -fv log_live_datasource_*.csv*
rm -fv log_master_run*txt
rm -fv log_try_all_tests*.txt
rm -fv plt_live_trainer_*pdf
rm -fv ckpt.tar
DIRS="datasources qlearners sim tests trainers utils"
for dir in $DIRS
do
    rm -rf ${dir}/__pycache__
done

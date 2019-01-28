#!/bin/bash
rm -f log_engine_*.csv*
rm -f tmplog.csv*
rm -f log_live_datasource_*.csv*
rm -f log_master_run*txt
rm -f plt_live_trainer_*pdf
DIRS="datasources qlearners sim tests trainers utils"
for dir in $DIRS
do
    rm -rf ${dir}/__pycache__
done

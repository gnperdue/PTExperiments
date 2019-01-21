#!/bin/bash
rm -f log_engine_*.csv*
rm -f tmplog.csv*
DIRS="datasources qlearners sim tests trainers utils"
for dir in $DIRS
do
    rm -rf ${dir}/__pycache__
done

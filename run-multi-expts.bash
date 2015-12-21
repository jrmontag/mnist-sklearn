#!/usr/bin/env bash

# vars
VENV=tmp-venv
PY=${VENV}/bin/python


echo "$(date +%Y-%m-%d\ %H:%M:%S) -- started running $0"


#for i in `seq 4 21`; do
for i in `seq 4 6`; do
    echo "$(date +%Y-%m-%d\ %H:%M:%S) -- launching experiment ${i}"
    ${PY} run-experiment.py --verbose expt_${i} > log/2015-12-20T035700_expt_${i}.log & 
    echo "$(date +%Y-%m-%d\ %H:%M:%S) -- sleeping"
    sleep 5
done

echo "$(date +%Y-%m-%d\ %H:%M:%S) -- finished"


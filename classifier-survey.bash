#!/usr/bin/env bash

#
# This script is used to execute a broad survey of classifiers  
#   (experiment pipelines #4-21) all with mostly default settings, 
#   as defined in  models.py  
#

# vars
VENV=tmp-venv
PY=${VENV}/bin/python

# work-around for ubuntu
PY=python


echo "$(date +%Y-%m-%d\ %H:%M:%S) -- started running $0"


for i in `seq 4 21`; do
    echo "$(date +%Y-%m-%d\ %H:%M:%S) -- launching experiment ${i}"
    ${PY} run-experiment.py --verbose expt_${i} > log/2015-12-20T035700_expt_${i}.log & 
    echo "$(date +%Y-%m-%d\ %H:%M:%S) -- sleeping for 60 seconds"
    sleep 60 
done

echo "$(date +%Y-%m-%d\ %H:%M:%S) -- finished launching experiments"


#!/usr/bin/env bash

#
# This script is used to launch a set of experiments or processes 
#

# vars
#VENV=tmp-venv
#PY=${VENV}/bin/python

# manual work-around for ubuntu system install
PY=python
# necessary once we include MLP - dev release of sklearn in virtual env
source /home/jmontague/CCC-venv/bin/activate
# I think we don't need this anymore b/c of symlink
#export PYTHONPATH=~/CCC-venv/lib/python2.7/site-packages/scikit-learn:$PYTHONPATH



####################  config  #############################
# range of experiments to run 
#SEQUENCE=`seq 28 29`
SEQUENCE=42

# are we doing a cv split ("experiment") or full test prediction?
EXPERIMENT=false

# server or laptop?
SERVER=true

# expanded (or original) data?
EXPANDED=true


# stagger the workload 
# 300 = 5 min
# 600 = 10 min
# 1200 = 20 min
SLEEPTIME=900
###########################################################


# remember bash boolean conditionals are confusing; just 
#   use convenient strings
# http://stackoverflow.com/a/21210966/1851811
if [ "${EXPERIMENT}" = true ]; then
    SCRIPT=run-experiment.py
    # be polite 
    NICE="nice -n15"
else
    SCRIPT=full-train-and-predict.py
    # be slightly less polite 
    NICE="nice -n5"
fi

if [ "${SERVER}" = true ]; then
    ARGS="--verbose --ubuntu"
else
    ARGS="--verbose"
fi

if [ "${EXPANDED}" = true ]; then
    ARGS="${ARGS} --expanded"
fi


echo "$(date +%Y-%m-%d\ %H:%M:%S) -- started running $0"
echo "$(date +%Y-%m-%d\ %H:%M:%S) -- using python interpreter: $(which python)"

# name all of these processes similarly 
filedate="$(date +%Y-%m-%dT%H:%M:%S)"

for i in ${SEQUENCE}; do
    echo "$(date +%Y-%m-%d\ %H:%M:%S) -- launching experiment ${i} with ${SCRIPT} and ARGS=${ARGS}"
    # launch the appropriate process 
    nohup ${NICE} ${PY} ${SCRIPT} expt_${i} ${ARGS} > log/${filedate}_expt_${i}.log & 
    # note this will also sleep after the last process
    echo "$(date +%Y-%m-%d\ %H:%M:%S) -- sleeping for ${SLEEPTIME} seconds"
    sleep ${SLEEPTIME} 
done


echo "$(date +%Y-%m-%d\ %H:%M:%S) -- deactivating virtualenv"
deactivate

echo "$(date +%Y-%m-%d\ %H:%M:%S) -- interpreter now set to: $(which python)"
echo "$(date +%Y-%m-%d\ %H:%M:%S) -- finished launching experiments"

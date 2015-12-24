#!/usr/bin/env bash

#
# This script is used to launch a set of experiments or processes 
#

# vars
VENV=tmp-venv
PY=${VENV}/bin/python
# manual work-around for ubuntu
PY=python


####################  config  #############################
# range of experiments to run 
SEQUENCE=`seq 33 34`
#SEQUENCE=32

# are we doing a cv split ("experiment") or full test prediction?
EXPERIMENT=true 

# server or laptop?
SERVER=true

# stagger the workload 
SLEEPTIME=60
###########################################################


# remember bash boolean conditionals are confusing; just 
#   use convenient strings
# http://stackoverflow.com/a/21210966/1851811
if [ "${EXPERIMENT}" = true ]; then
    SCRIPT=run-experiment.py
else
    SCRIPT=full-train-and-predict.py
fi

if [ ${SERVER} = "true" ]; then
    ARGS='--verbose --ubuntu'
else
    ARGS='--verbose'
fi


# be polite 
NICE="nice -n15"


echo "$(date +%Y-%m-%d\ %H:%M:%S) -- started running $0"

filedate="$(date +%Y-%m-%dT%H:%M:%S)"

for i in ${SEQUENCE}; do
    echo "$(date +%Y-%m-%d\ %H:%M:%S) -- launching experiment ${i}"
    nohup ${NICE} ${PY} ${SCRIPT} expt_${i} ${ARGS} > log/${filedate}_expt_${i}.log & 
    echo "$(date +%Y-%m-%d\ %H:%M:%S) -- sleeping for ${SLEEPTIME} seconds"
    sleep ${SLEEPTIME} 
done

echo "$(date +%Y-%m-%d\ %H:%M:%S) -- finished launching experiments"



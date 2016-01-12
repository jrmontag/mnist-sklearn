#!/usr/bin/env bash

#
# This script is used to launch a set of experiments or processes 
#

####################  config  #############################
# range of experiments to run 
SEQUENCE=`seq 1 3`
#SEQUENCE=42

# are we doing a cv split ("experiment")? (vs. train and predict on test) 
EXPERIMENT=true

# server (ubuntu)? (vs. laptop)
SERVER=true

# expanded dataset? (vs. as-given)
EXPANDED=false

# stagger the workload (seconds)
SLEEPTIME=10
# 60 = 1 min
# 300 = 5 min
# 600 = 10 min
# 1200 = 20 min
###########################################################

## variables
# log dir
LOGDIR=log
# virtualenv
#VENV=/home/jmontague/CCC-venv
VENV=tmp-venv
#VENV=dev-tmp-venv
# Python script
PY_SCRIPT=build-model.py

# activate the appropriate virtualenv
source ${VENV}/bin/activate 

# set some defaults
ARGS="--verbose"
NICE="nice -n10"


# remember bash boolean conditionals are confusing; just 
#   use convenient strings
# http://stackoverflow.com/a/21210966/1851811
if [ "${EXPERIMENT}" = true ]; then
    # be polite 
    NICE="nice -n15"
else
    # be slightly less polite 
    NICE="nice -n5"
    ARGS="${ARGS} --submission"
fi

if [ "${SERVER}" = true ]; then
    ARGS="${ARGS} --ubuntu"
fi

if [ "${EXPANDED}" = true ]; then
    ARGS="${ARGS} --expanded"
fi


echo "$(date +%Y-%m-%d\ %H:%M:%S) -- started running $0"
echo "$(date +%Y-%m-%d\ %H:%M:%S) -- using python interpreter: $(which python)"

# name all of these processes similarly 
filedate="$(date +%Y-%m-%dT%H:%M:%S)"

for i in ${SEQUENCE}; do
    echo "$(date +%Y-%m-%d\ %H:%M:%S) -- launching experiment ${i} with ${PY_SCRIPT} and ARGS=${ARGS}"
    # launch the appropriate process 
    #   - run this bash script w/ nohup & all of the python procs will inherit it 
    ${NICE} python ${PY_SCRIPT} expt_${i} ${ARGS} > ${LOGDIR}/${filedate}_expt_${i}.log & 
    # note this will also sleep after the last process
    echo "$(date +%Y-%m-%d\ %H:%M:%S) -- sleeping for ${SLEEPTIME} seconds"
    sleep ${SLEEPTIME} 
done


echo "$(date +%Y-%m-%d\ %H:%M:%S) -- deactivating virtualenv"
deactivate

echo "$(date +%Y-%m-%d\ %H:%M:%S) -- interpreter now set to: $(which python)"
echo "$(date +%Y-%m-%d\ %H:%M:%S) -- finished launching experiments"

#!/usr/bin/env bash

#
# This script is used to launch a set of experiments or processes 
#

# vars
VENV=tmp-venv
PY=${VENV}/bin/python
# manual work-around for ubuntu
PY=python

# broad survey of classifiers 
SEQUENCE=`seq 4 21`
# broad survey of classifiers 
#SEQUENCE="4 6"

# experiments or submissions?
SCRIPT=run-experiment.py
#SCRIPT=full-train-and-predict.py

# for experiments
NICE="nice 15"
# for submissions
NICE="nice"

# survey (plotting)
#ARGS='--verbose --ubuntu'
# submission
ARGS='--verbose'

# kNN model: expt_4
# SVM mode: expt_6

SLEEPTIME=60


echo "$(date +%Y-%m-%d\ %H:%M:%S) -- started running $0"

filedate="$(date +%Y-%m-%dT%H:%M:%S)"

for i in ${SEQUENCE}; do
    echo "$(date +%Y-%m-%d\ %H:%M:%S) -- launching experiment ${i}"
    nohup ${NICE} ${PY} ${SCRIPT} expt_${i} ${ARGS} > log/${filedate}_expt_${i}.log & 
    echo "$(date +%Y-%m-%d\ %H:%M:%S) -- sleeping for ${SLEEPTIME} seconds"
    sleep ${SLEEPTIME} 
done

echo "$(date +%Y-%m-%d\ %H:%M:%S) -- finished launching experiments"


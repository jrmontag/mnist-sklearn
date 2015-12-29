#!/usr/bin/env python
# -*- coding: UTF-8 -*-
__author__="Josh Montague"
__license__="MIT License"

import argparse
from datetime import datetime
import logging
import os
import sys

import numpy as np
from sklearn.externals import joblib
from sklearn.cross_validation import cross_val_score, cross_val_predict, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from models import experiment_dict
import utils


parser = argparse.ArgumentParser()
parser.add_argument('expt', help='specify experiment to run (see README)')
parser.add_argument('--expanded', action='store_true',
                    help='read expanded (perturbation) data files')
parser.add_argument('--ubuntu', action='store_true',
                    help='modify imports if running on ubuntu')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='increase output verbosity')
args = parser.parse_args()


# use a simple logger - get the level from the cmd line
loglevel = logging.DEBUG if args.verbose else logging.INFO
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', 
                    stream=sys.stdout, level=loglevel)
logging.debug('logging enabled - preparing for experiment') 

# read the right files
files='original'
if args.expanded:
    files='expanded' 

logging.info('reading {} input data from disk'.format(files)) 
try:
    X_train_full, y_train_full, X_test = utils.load_np_arrays(files) 
    logging.debug('observed data dimensions: {}, {}. {}'.format(
                    X_train_full.shape, y_train_full.shape, X_test.shape))
except IOError, e:
    # let it crash, but give some insight in the log 
    logging.warn('Error reading data from files (do they exist yet?)') 
    logging.warn('Error message={}'.format(e)) 
    raise e

# get pipeline 
pipeline_detail = experiment_dict[args.expt]
logging.debug('reading specified experiment pipeline={}'.format(pipeline_detail)) 
pipeline = pipeline_detail['pl']

# fit model on entire training dataset
logging.info('fitting model on entirety of training data') 
logging.debug('dataset sizes (xtrain_full, xtest, ytrain_full)={}'.format(
                        [i.shape for i in [X_train_full, X_test, y_train_full]]))
pipeline.fit(X_train_full, y_train_full)

# store the full model 
model_name = utils.short_name(pipeline) + \
                '_full-data_' + \
                datetime.utcnow().strftime('%Y-%m-%d_%H%M%S') 
logging.info('writing fit pipeline to disk as {}'.format(model_name)) 
try:
    joblib.dump(pipeline, os.path.join('saved_models', model_name) + '.pkl', compress=3)
except OverflowError, e:
    # this is annoying; look deeper, later
    logging.warn('joblib write failed with error={}'.format(e)) 
    logging.info('proceeding with predictions without writing model to disk')

# predict 
logging.info('predicting test data')
predictions = pipeline.predict(X_test)

# submission file 
logging.info('writing predictions to formatted submission file')
utils.create_submission(predictions, 
                        pipeline_detail['name'], 
                        comment=pipeline_detail['note']
                        )

logging.info('completed pipeline={}'.format(pipeline_detail))



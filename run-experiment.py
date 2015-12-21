#!/usr/bin/env python
# -*- coding: UTF-8 -*-
__author__="Josh Montague"
__license__="MIT License"

import argparse
from datetime import datetime
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from sklearn.externals import joblib
from sklearn.cross_validation import cross_val_score, cross_val_predict, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from models import experiment_dict
#from utils import name, create_submission, load_np_arrays
import utils


parser = argparse.ArgumentParser()
parser.add_argument('expt', help='specify experiment to run (see README)', 
                    choices=['expt_1', 'expt_2', 'expt_3', 'expt_4', \
                        'expt_5', 'expt_6', 'expt_7', 'expt_8', 'expt_9', \
                        'expt_10', 'expt_11', 'expt_12', 'expt_13', \
                        'expt_14', 'expt_15', 'expt_16', 'expt_17', \
                        'expt_18', 'expt_19', 'expt_20'])
parser.add_argument('-v', '--verbose', action='store_true',
                    help='increase output verbosity')
args = parser.parse_args()


# use a simple logger - get the level from the cmd line
loglevel = logging.DEBUG if args.verbose else logging.INFO
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', 
                    stream=sys.stdout, level=loglevel)
logging.debug('logging enabled - preparing for experiment') 


logging.info('reading input data from disk') 
try:
    X_train_full, y_train_full, X_test = utils.load_np_arrays() 
    logging.debug('observed data dimensions: {}, {}. {}'.format(
                    X_train_full.shape, y_train_full.shape, X_test.shape))
except IOError, e:
    # let it crash, but give some insight in the log 
    logging.warn('Error reading data from files (do they exist yet?)') 
    logging.warn('Error message={}'.format(e)) 
    raise e

# get pipeline 
logging.debug('reading specified experiment from models') 
pipeline_detail = experiment_dict[args.expt]
logging.debug('specified experiment pipeline={}'.format(pipeline_detail)) 
pipeline = pipeline_detail['pl']

# train/test split
train_fraction = 0.8
logging.info('creating train/test split with train% = {}'.format(train_fraction)) 
X_train, X_test, y_train, y_test = train_test_split(X_train_full, 
                                                    y_train_full,
                                                    test_size=train_fraction, 
                                                    random_state=42) 
logging.debug('dataset sizes (xtrain, xtest, ytrain, ytest)={}'.format(
                        [i.shape for i in [X_train, X_test, y_train, y_test]]))

# fit
logging.info('fitting experiment pipeline with signature={}'.format(pipeline)) 
pipeline.fit(X_train, y_train)

# store the train/split version 
model_name = utils.short_name(pipeline) + \
                '_cv-split_' + \
                datetime.utcnow().strftime('%Y-%m-%d_%H%M%S') 
logging.info('writing fit pipeline to disk as {}'.format(model_name)) 
joblib.dump(pipeline, os.path.join('saved_models', model_name) + '.pkl', compress=3)

# cv for accuracy 
#
# this gives a better idea of uncertainty, but it adds 'cv' more
#   fits of the model - hold onto it for later 
#
cv = 3
logging.info('cross validating model accuracy with cv={}'.format(cv))
scores = cross_val_score(pipeline, X_test, y_test, cv=cv) 
logging.info('obtained accuracy={:0.2f}% +/- {:0.2f} with cv={}, \
                pipeline={} '.format(scores.mean()*100, 
                                    scores.std()*100*2, 
                                    cv, 
                                    pipeline))

# cv for predictions 
logging.info('cross validating model predictions with cv={}'.format(cv))
predictions = cross_val_predict(pipeline, X_test, y_test, cv=cv)
logging.info('obtained accuracy = {:.2f}% with cv={}, pipeline={} '.format(
                                        accuracy_score(y_test,predictions)*100,
                                        cv,
                                        pipeline
                                        )) 

# create confusion matrix + save figure 
# > TODO: move this figure creation into utils + add cv accuracy to title) <
logging.info('calculating confusion matrix')
sb.heatmap(confusion_matrix(y_test, predictions))
plt.title(model_name + ' ({:.2f}%)'.format(scores.mean()*100))
plt.xlabel("True") 
plt.ylabel("Pred")
#plt.tight_layout()
logging.info('saving confusion matrix')
plt.savefig(os.path.join('saved_models', model_name) + '.pdf'
            , format='pdf', bbox_inches='tight')
                            
logging.info('completed pipeline={}'.format(pipeline_detail))



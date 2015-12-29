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
#from utils import name, create_submission, load_np_arrays
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

# ubuntu v. os x shenanigans
if args.ubuntu:
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sb


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
# if this was a gridsearch, log the combinations (and call out the winner) 
if hasattr(pipeline, 'best_params_'):
    logging.info('best gridsearch score={}, best set of pipeline params = {}'.format(
                                                                                pipeline.best_score_, 
                                                                                pipeline.best_params_)) 
    logging.info('now displaying all pipeline param scores')
    for params, mean_score, scores in pipeline.grid_scores_:
        logging.info("{:0.3f} (+/-{:0.03f}) for {}".format(mean_score, scores.std() * 2, params))

# store the train/split version 
model_name = utils.short_name(pipeline) + \
                '_cv-split_' + \
                datetime.utcnow().strftime('%Y-%m-%d_%H%M%S') 
logging.info('writing fit pipeline to disk as {}'.format(model_name)) 
try:
    joblib.dump(pipeline, os.path.join('saved_models', model_name) + '.pkl', compress=3)
except OverflowError, e:
    # this is annoying; look deeper, later
    logging.warn('joblib write failed with error={}'.format(e)) 
    logging.info('proceeding with predictions without writing model to disk')

# if we used a gridsearch, this is simple 
if hasattr(pipeline, 'best_params_'):
    logging.info('predicting test values with best-choice gridsearch params')
    # fake an array of CV scores to play nice with plot formatting later 
    scores = np.array([pipeline.best_score_])
    predictions = pipeline.predict(X_test) 
else:
    # otherwise, run a cross-validation for test accuracy
    # cv for accuracy 
    cv = 3
    # this gives a better idea of uncertainty, but it adds 'cv' more
    #   fits of the model 
#    #### add an args.cross_val_score to turn this off
#    logging.info('cross validating model accuracy with cv={}'.format(cv))
#    scores = cross_val_score(pipeline, X_test, y_test, cv=cv) 
#    logging.info('obtained accuracy={:0.2f}% +/- {:0.2f} with cv={}, \
#                    pipeline={} '.format(scores.mean()*100, 
#                                        scores.std()*100*2, 
#                                        cv, 
#                                        pipeline))

    # cv for predictions 
    logging.info('cross validating model predictions with cv={}'.format(cv))
    predictions = cross_val_predict(pipeline, X_test, y_test, cv=cv)
    logging.info('obtained accuracy = {:.2f}% with cv={}, pipeline={} '.format(
                                            accuracy_score(y_test,predictions)*100,
                                            cv,
                                            pipeline
                                            )) 

# create confusion matrix + save figure 
# > TODO: move this figure creation into utils <
logging.info('calculating confusion matrix')
sb.heatmap(confusion_matrix(y_test, predictions))
try:
    plt.title(model_name + ' [expt] ({:.2f}%)'.format(scores.mean()*100))
except NameError:
    logging.debug('didnt find "scores" from cross_val_score, calculating accuracy by accuracy_score()')
    plt.title(model_name + ' [expt] ({:.2f}%)'.format(accuracy_score(y_test,predictions)*100))
plt.xlabel("True") 
plt.ylabel("Pred")
#plt.tight_layout()
logging.info('saving confusion matrix')
plt.savefig(os.path.join('saved_models', model_name) + '.pdf'
            , format='pdf', bbox_inches='tight')
                            
logging.info('completed pipeline={}'.format(pipeline_detail))



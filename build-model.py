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
parser.add_argument('--submission', action='store_true',
                    help='train model on all of data + create submission file')
parser.add_argument('--cross_val_score', action='store_true',
                    help='run addl cross val score for uncertainty in accuracy (costly!)')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='increase output verbosity')
args = parser.parse_args()


# only import mpl when necessary (not a submission)
if not args.submission:
    if args.ubuntu: 
        # ubuntu v. os x shenanigans 
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sb

# use a simple logger - get the level from the cmd line
loglevel = logging.DEBUG if args.verbose else logging.INFO
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', 
                    stream=sys.stdout, level=loglevel)

# what are we running?
job = 'submission' if args.submission else 'experiment'
logging.debug('logging enabled for this {}'.format(job)) 

# read the right files
files = 'expanded' if args.expanded else 'original'
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

# get appropriate pipeline + metadata 
pipeline_detail = experiment_dict[args.expt]
logging.debug('specified experiment pipeline={}'.format(pipeline_detail)) 
pipeline = pipeline_detail['pl']

if args.submission:
    # making a submission; train on all given data
    logging.info('fitting models to entire training set')
    X_train, y_train = X_train_full, y_train_full
else:
    # running an experiment - cross validate with train/test split
    train_fraction = 0.8
    logging.info('fitting models to cv train/test split with train% = {}'.format(train_fraction)) 
    X_train, X_test, y_train, y_test = train_test_split(X_train_full, 
                                                        y_train_full,
                                                        test_size=train_fraction, 
                                                        random_state=42) 
logging.debug('fitting model to array sizes (xtrain, ytrain)={}'.format(
                                        [i.shape for i in [X_train, y_train]]))
    
# fit
logging.info('fitting experiment pipeline with signature={}'.format(pipeline)) 
pipeline.fit(X_train, y_train)

if args.submission:
    fname_spec = '_submission_'
else: 
    # gridsearch? log all results + call out the winner 
    if hasattr(pipeline, 'best_params_'):
        logging.info('best gridsearch score={}'.format(pipeline.best_score_)) 
        logging.info('best set of pipeline params={}'.format(pipeline.best_params_)) 
        logging.info('now displaying all pipeline param scores...')
        for params, mean_score, scores in pipeline.grid_scores_:
            logging.info("{:0.3f} (+/-{:0.03f}) for {}".format(mean_score, scores.std()*2, params))
    fname_spec = '_expt_' 

# build proper file name so we can reference it in the logs 
model_name = utils.short_name(pipeline) + \
                fname_spec + \
                datetime.utcnow().strftime('%Y-%m-%d_%H%M%S') 

logging.info('writing fit {} pipeline to disk as {}'.format(job, model_name)) 
try:
    joblib.dump(pipeline, os.path.join('saved_models', model_name) + '.pkl', compress=3)
except OverflowError, e:
    # this is annoying; look into it later 
    logging.warn('joblib write failed with error={}'.format(e)) 
    logging.info('proceeding with predictions without writing model to disk')

# do something useful with the fit model
if args.submission:
    # make predictions for a leaderboard submission
    logging.info('writing predictions to formatted submission file')
    utils.create_submission(predictions, 
                            pipeline_detail['name'], 
                            comment=pipeline_detail['note'])
else:
    # if we already did CV through the gridsearch, then just take 
    #   the best score and make predictions  
    if hasattr(pipeline, 'best_params_'):
        logging.info('predicting test values with best-choice gridsearch params')
        predictions = pipeline.predict(X_test) 
        # fake an array of CV scores to play nice with plot formatting later 
        scores = np.array([pipeline.best_score_])
    # otherwise, do some cross validation
    else:
        # otherwise, run a cross-validation for test accuracy
        cv = 3
        logging.info('cross validating model predictions with cv={}'.format(cv))
        predictions = cross_val_predict(pipeline, X_test, y_test, cv=cv)
        logging.info('obtained accuracy = {:.2f}% with cv={}, pipeline={} '.format(
                                                accuracy_score(y_test,predictions)*100,
                                                cv,
                                                pipeline)) 
        if args.cross_val_score:
            # this gives a better idea of uncertainty, but it adds 'cv' more
            #   fits to the pipeline (expensive!)
            logging.info('cross val score flag found')
            logging.info('cross validating model accuracy with cv={}'.format(cv))
            scores = cross_val_score(pipeline, X_test, y_test, cv=cv) 
            logging.info('obtained accuracy={:0.2f}% +/- {:0.2f} with cv={}, \
                                        pipeline={} '.format(scores.mean()*100, 
                                                            scores.std()*100*2, 
                                                            cv, 
                                                            pipeline))
        # if running an experiment, plot confusion matrix for review 
        # > TODO: move this figure creation into utils <
        logging.info('calculating confusion matrix')
        try:
            sb.heatmap(confusion_matrix(y_test, predictions))
        except RuntimeError, e:
            logging.warn('plotting error. matplotlib backend may need to be changed (see readme). error={}'.format(e))
            logging.warn('plot may still have been saved, and model has already been saved to disk.')
        try:
            plt.title(model_name + ' [expt] ({:.2f}%)'.format(scores.mean()*100))
        except NameError:
            logging.debug('didnt find "scores" from cross_val_score, calculating accuracy by accuracy_score()')
            plt.title(model_name + ' [expt] ({:.2f}%)'.format(accuracy_score(y_test,predictions)*100))
        plt.xlabel("True") 
        plt.ylabel("Pred")
        #plt.tight_layout()
        logging.info('saving confusion matrix')
        plt.savefig(os.path.join('saved_models', model_name) + '.pdf',
                                    format='pdf', 
                                    bbox_inches='tight')
                            
logging.info('completed {} with pipeline={}'.format(job, pipeline_detail))


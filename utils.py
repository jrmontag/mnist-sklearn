# -*- coding: UTF-8 -*-
__author__="Josh Montague"
__license__="MIT License"

#
# This module defines a number of helper functions. 
#

from datetime import datetime
import logging
import numpy as np
import os
import sys


# set up a logger
util_logr = logging.getLogger(__name__)
util_logr.setLevel(logging.DEBUG)
util_sh = logging.StreamHandler(stream=sys.stdout)
formatter = logging.Formatter('%(asctime)s : %(name)s : %(levelname)s : %(message)s')
util_sh.setFormatter(formatter)
util_logr.addHandler(util_sh)


def short_name(model):
    """Return a simplified name for this model. A bit brittle."""
    # for a single model, this will work
    name = model.__class__.__name__
    try:
        if hasattr(model, 'steps'):
            # pipeline
            name = '-'.join( [ pair[0] for pair in model.steps ] )
        elif hasattr(model, 'best_estimator_'):
            if hasattr(model.estimator, 'steps'):
                # gridsearchcv
                name = 'gscv_' + '-'.join( [x[0] for x in model.estimator.steps ])
            elif hasattr(model.estimator, 'estimators'):
                # votingclassifier
                name = 'gscv_vc_' + '-'.join( [x[0] for x in model.estimator.estimators ])
        elif hasattr(model, 'base_estimator_'):
            # bagging
            name = 'bag_' + short_name(model.base_estimator)
    except AttributeError, e:
        util_logr.info('utils.short_name() couldnt generate quality name')
        # for a single model, this will work
        name = model.__class__.__name__
        util_logr.info('falling back to generic name={}'.format(name))
    return name 


def create_submission(predictions, sub_name, comment=None, team='DrJ'):
    """Include the specified array of image predictions in a 
    properly-formatted submission file.
    """
    now = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')
    submission_name = '-'.join(sub_name.split())
    with open('submissions/{}_{}.submission'.format(
            now, submission_name), 'w') as f:

        f.write('#'*20 + ' Generated submission file\n')
        if comment is not None:
            f.write('# ' + comment + '\n')
        f.write('{}\n'.format(team))
        f.write('{}\n'.format(now))
        f.write('{}\n'.format(sub_name))
        for p in predictions:
            f.write('{}\n'.format(p))
    return True


def load_np_arrays(files='original'):
    """
    Return numpy arrays for training dataset, training labels, 
    and test dataset (in that order). If files='original', 
    return the image data, as given. If files='expanded', 
    return the perturbed image files (~5 times larger). 
    
    files='original', 'expanded' 
    """
    # nb: path assumes that we call this function from project root  
    train_imgs_f = 'train-images.npy'
    train_labels_f = 'train-labels.npy'

    if files == 'expanded':
        train_imgs_f = 'expanded-' + train_imgs_f 
        train_labels_f = 'expanded-' + train_labels_f 

    X_train = np.load(os.path.join('data', train_imgs_f))
    y_train = np.load(os.path.join('data', train_labels_f))
    X_test = np.load(os.path.join('data', 'test-images.npy'))

    return (X_train, y_train, X_test)



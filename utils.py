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


def short_name(pipeline):
    """Return the simplified name of this model."""
    # for a single model, this works
    #return model.__class__.__name__
    # for a pipeline, we need a nice string version of the steps
    return '-'.join( [ pair[0] for pair in pipeline.steps ] )


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


def load_np_arrays():
    """Return numpy arrays for training dataset, training labels, 
    and test dataset (in that order). 
    """
    # nb: path assumes that we call this function from project root  
    #   - may not be appropriate for importing into code in bin/ ?
    X_train = np.load(os.path.join('data', 'train-images.npy'))
    y_train = np.load(os.path.join('data', 'train-labels.npy'))
    X_test = np.load(os.path.join('data', 'test-images.npy'))

    return (X_train, y_train, X_test)



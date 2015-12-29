#!/usr/bin/env python
# -*- coding: UTF-8 -*-
__author__="Josh Montague"
__license__="MIT License"

"""
Adapted from: 
https://github.com/mnielsen/neural-networks-and-deep-learning
"""

import argparse
import gc
import logging
import numpy as np
import random
import sys

import utils


# this will be helpful for displaying arrays
np.set_printoptions(linewidth=200)

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--verbose", action="store_true",
                                  help="increase output verbosity")
args = parser.parse_args()

# use a simple logger - get the level from the cmd line
loglevel = logging.DEBUG if args.verbose else logging.INFO
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', 
                    stream=sys.stdout, level=loglevel)
logging.debug('logging enabled - preparing for work')


logging.info('reading original input data from disk') 
try:
    # nb: we don't need X_test right now
    X_train_full, y_train_full, X_test = utils.load_np_arrays() 
    logging.debug('observed data dimensions: {}, {}. {}'.format(
                    X_train_full.shape, y_train_full.shape, X_test.shape))
except IOError, e:
    # let it crash, but give some insight in the log 
    logging.warn('Error reading data from files (do they exist yet?)') 
    logging.warn('Error message={}'.format(e)) 
    raise e

# define the perturbations
perturb_list = [
    # displacement, axis, index position, index
    (1,  0, "first", 0),
    (-1, 0, "first", 27),
    (1,  1, "last",  0),
    (-1, 1, "last",  27)]

# a container for our new data set
expanded_data = []

# loop over all images in training set 
for i, (img, label) in enumerate(zip(X_train_full, y_train_full)):
    if i % 100 == 0:
        logging.info('perturbing image number {}, current length of expanded_data={}'.format(i, len(expanded_data)))
    # add the original array to the new array
    expanded_data.append((img, label)) 
    # reshape back into square for roll()
    img = np.reshape(img, (28,28))
    # perturb in each direction
    for d, ax, position, idx in perturb_list:
        #logging.debug('current perturb_list={}'.format([d,ax,position,idx]))
        # shift pixels in this image by d along ax
        perturbed_img = np.roll(img, d, ax)
        # in case pixels get shifted across the edge boundaries, 
        #   we can just set the corresponding edge to 0s (first 
        #   reshape the array so we can slice efficiently) 
        #perturbed_img = np.reshape(perturbed_img, (28,28))
        if position == "first": 
            # first row/column
            perturbed_img[idx, :] = np.zeros(28)
        else: 
            # last row/column
            perturbed_img[:, idx] = np.zeros(28)
        # add new (flattened) image and label to the expanded list 
        expanded_data.append( (np.reshape(perturbed_img, 784), label) )
        # e_d ~ [(np.arr, int), ... ]
        
        logging.debug('current pertub_list={}'.format([d,ax,position,idx]))
        logging.debug('current label={}'.format(label))
        logging.debug('original image array=\n{}'.format(img))
        logging.debug('shifted data array=\n{}'.format(perturbed_img))


# shuffle to avoid bias in array positions
logging.info('shuffling expanded data set')
random.shuffle(expanded_data)
logging.debug('expanded_data=\n{}'.format(expanded_data))

# e_d is a list of (img-array, label) tuples 
#   - zip(*e_d) pairs the elements of each img-array  
logging.info('converting expanded data to list of numpy arrays')
expanded_data_array_list = [np.array(x) for x in zip(*expanded_data)] 
logging.info('length of expanded array list={}'.format(len(expanded_data_array_list))) 

# extract the labels from the last column of the array 
#y_expanded = expanded_data_array[:,-1]
# and the data from everything *but* the last column 
#X_expanded = expanded_data_array[:,:-1]
X_expanded = expanded_data_array_list[0]
y_expanded = expanded_data_array_list[1]

logging.debug('X_expanded (length={})=\n{}'.format(len(X_expanded), X_expanded))
logging.debug('y_expanded (length={})=\n{}'.format(len(y_expanded), y_expanded))

# verify
n=5
for i in range(n):
    logging.debug('expanded image {} (label={}):\n{}'.format(i, y_expanded[i], np.reshape(X_expanded[i,:], (28,28))))


# after the for loop, save for later use (more transparently, with .npy arrays) 
for name, dataset in [('expanded-train-images', X_expanded) , ('expanded-train-labels', y_expanded)]:
    logging.info("writing {} to disk as numpy array".format(name)) 
    with open('data/{}.npy'.format(name), 'wb') as f:
        np.save(f, dataset)

logging.info('done expanding data')


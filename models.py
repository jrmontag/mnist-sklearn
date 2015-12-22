#!/usr/bin/env python
# -*- coding: UTF-8 -*-
__author__="Josh Montague"
__license__="MIT License"

# this module defines models and pipelines for import into 
#   individual experiment runs 

import logging
import sys

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import SGDClassifier 
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


from sklearn.pipeline import Pipeline


experiment_dict = \
    { 
    # experiments to build pipeline
    'expt_1': { 
        'note': 'random guessing (maintains class distributions)',
        'name': 'Crash Test Dummies', 
        'pl': Pipeline([ ('dummy_clf', DummyClassifier()) ])
        },

    'expt_2': { 
        'note': 'vanilla linear svm (heard it through the grapevine)',
        'name': 'Grapevine',
        'pl': Pipeline([ ('linear_svm', SGDClassifier(n_jobs=-1)) ]) 
        },
    'expt_3': { 
        'note': 'add scaling prior to SVM (you must be this tall to ride)',
        'name': 'This tall to ride',
        'pl': Pipeline([ ('scaling', StandardScaler()), ('linear_svm', SGDClassifier(n_jobs=-1)) ]) 
        },
    # systematic check of all classifiers + scaling
    'expt_4': { 
        'note': 'vanilla knn (mac and kelly from 2014 "neighbors"',
        'name': 'Mac and Kelly',
        'pl': Pipeline([ ('knn', KNeighborsClassifier(n_jobs=-1)) ]) 
        },
    'expt_5': { 
        'note': 'scaled knn',
        'name': 'scaled knn',
        'pl': Pipeline([ ('scaling', StandardScaler()), ('knn', KNeighborsClassifier(n_jobs=-1)) ]) 
        },
    'expt_6': { 
        'note': 'rbf kernel SVM', 
        'name': 'Popcorn machine',
        'pl': Pipeline([ ('rbf-svm', SVC(kernel='rbf')) ]) 
        },
    'expt_7': { 
        'note': 'scaled rbf kernel SVM',
        'name': 'scaled rbf kernel SVM', 
        'pl': Pipeline([ ('scaling', StandardScaler()), ('rbf-svm', SVC(kernel='rbf')) ]) 
        },
    'expt_8': { 
        'note': 'default decision tree',
        'name': 'default decision tree',
        'pl': Pipeline([ ('decision-tree', DecisionTreeClassifier()) ]) 
        },
    'expt_9': { 
        'note': 'scaled default decision tree',
        'name': 'scaled default decision tree',
        'pl': Pipeline([ ('scaling', StandardScaler()), ('decision-tree', DecisionTreeClassifier()) ]) 
        },
    'expt_10': { 
        'note': 'default RF',
        'name': 'default RF',
        'pl': Pipeline([ ('random-forest', RandomForestClassifier()) ]) 
        },
    'expt_11': { 
        'note': 'scaled default RF',
        'name': 'scaled default RF',
        'pl': Pipeline([ ('scaling', StandardScaler()), ('random-forest', RandomForestClassifier()) ]) 
        },
    'expt_12': { 
        'note': 'default adaboost',
        'name': 'default adaboost',
        'pl': Pipeline([ ('DT-adaboost', AdaBoostClassifier()) ]) 
        },
    'expt_13': { 
        'note': 'scaled default adaboost',
        'name': 'scaled default adaboost',
        'pl': Pipeline([ ('scaling', StandardScaler()), ('DT-adaboost', AdaBoostClassifier()) ]) 
        },
    'expt_14': { 
        'note': 'default Gaussian NB',
        'name': 'default Gaussian NB',
        'pl': Pipeline([ ('gaussian-nb', GaussianNB()) ]) 
        },
    'expt_15': { 
        'note': 'scaled Gaussian NB',
        'name': 'scaled Gaussian NB',
        'pl': Pipeline([ ('scaling', StandardScaler()), ('gaussian-nb', GaussianNB()) ]) 
        },
    'expt_16': { 
        'note': 'default Multinomial NB',
        'name': 'default Multinomial NB',
        'pl': Pipeline([ ('multi-nb', MultinomialNB()) ]) 
        },
    'expt_17': { 
        'note': 'scaled Multinomial NB',
        'name': 'scaled Multinomial NB',
        'pl': Pipeline([ ('scaling', StandardScaler()), ('multi-nb', MultinomialNB()) ]) 
        },
    'expt_18': { 
        'note': 'default LDA',
        'name': 'default LDA',
        'pl': Pipeline([ ('linear-da', LinearDiscriminantAnalysis()) ]) 
        },
    'expt_19': { 
        'note': 'scaled LDA',
        'name': 'scaled LDA',
        'pl': Pipeline([ ('scaling', StandardScaler()), ('linear-da', LinearDiscriminantAnalysis()) ]) 
        },
    'expt_20': { 
        'note': 'default QDA',
        'name': 'default QDA',
        'pl': Pipeline([ ('Quadratic-da', QuadraticDiscriminantAnalysis()) ]) 
        },
    'expt_21': { 
        'note': 'scaled QDA',
        'name': 'scaled QDA',
        'pl': Pipeline([ ('scaling', StandardScaler()), ('Quadratic-da', QuadraticDiscriminantAnalysis()) ]) 
        }

    }



#!/usr/bin/env python
# -*- coding: UTF-8 -*-
__author__="Josh Montague"
__license__="MIT License"

# this module defines models and pipelines for import into 
#   individual experiment runs 

import logging
import numpy as np
import sys

# set up a logger, at least for the ImportError 
model_logr = logging.getLogger(__name__)
model_logr.setLevel(logging.DEBUG)
model_sh = logging.StreamHandler(stream=sys.stdout)
formatter = logging.Formatter('%(asctime)s : %(name)s : %(levelname)s : %(message)s')
model_sh.setFormatter(formatter)
model_logr.addHandler(model_sh)


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier, VotingClassifier 
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.manifold import TSNE
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
try:
    from sklearn.neural_network import MLPClassifier
except ImportError, e:
    model_logr.info('couldnt import sklearn.neural_network') 
    model_logr.info('... as of the time of writing, this requires a build of the dev release (see README)') 
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.pipeline import Pipeline


experiment_dict = \
    { 
    # Note: keys are of the form expt_*, which are used to execute the 
    #   associated values of 'pl' keys     
    #
    # experiments to build pipeline ################################################
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
        'pl': Pipeline([ ('scaling', StandardScaler()), 
                        ('linear_svm', SGDClassifier(n_jobs=-1)) ]) 
        },
    # systematic check of default classifiers + scaling ################################
    'expt_4': { 
        'note': 'vanilla knn (mac and kelly from 2014 "neighbors"',
        'name': 'Mac and Kelly',
        'pl': Pipeline([ ('knn', KNeighborsClassifier(n_jobs=-1)) ]) 
        },
    'expt_5': { 
        'note': 'scaled knn',
        'name': 'scaled knn',
        'pl': Pipeline([ ('scaling', StandardScaler()), 
                        ('knn', KNeighborsClassifier(n_jobs=-1)) ]) 
        },
    'expt_6': { 
        'note': 'rbf kernel SVM', 
        'name': 'rbf kernel SVM', 
        'pl': Pipeline([ ('rbf-svm', SVC(kernel='rbf')) ]) 
        },
    'expt_7': { 
        'note': 'scaled rbf kernel SVM',
        'name': 'Portable popcorn machine',
        'pl': Pipeline([ ('scaling', StandardScaler()), 
                        ('rbf-svm', SVC(kernel='rbf', cache_size=1000)) ]) 
        },
    'expt_8': { 
        'note': 'default decision tree',
        'name': 'default decision tree',
        'pl': Pipeline([ ('decision-tree', DecisionTreeClassifier()) ]) 
        },
    'expt_9': { 
        'note': 'scaled default decision tree',
        'name': 'scaled default decision tree',
        'pl': Pipeline([ ('scaling', StandardScaler()), 
                        ('decision-tree', DecisionTreeClassifier()) ]) 
        },
    'expt_10': { 
        'note': 'default RF',
        'name': 'default RF',
        'pl': Pipeline([ ('random-forest', RandomForestClassifier()) ]) 
        },
    'expt_11': { 
        'note': 'scaled default RF',
        'name': 'scaled default RF',
        'pl': Pipeline([ ('scaling', StandardScaler()), 
                        ('random-forest', RandomForestClassifier()) ]) 
        },
    'expt_12': { 
        'note': 'default adaboost',
        'name': 'default adaboost',
        'pl': Pipeline([ ('DT-adaboost', AdaBoostClassifier()) ]) 
        },
    'expt_13': { 
        'note': 'scaled default adaboost',
        'name': 'scaled default adaboost',
        'pl': Pipeline([ ('scaling', StandardScaler()), 
                        ('DT-adaboost', AdaBoostClassifier()) ]) 
        },
    'expt_14': { 
        'note': 'default Gaussian NB',
        'name': 'default Gaussian NB',
        'pl': Pipeline([ ('gaussian-nb', GaussianNB()) ]) 
        },
    'expt_15': { 
        'note': 'scaled Gaussian NB',
        'name': 'scaled Gaussian NB',
        'pl': Pipeline([ ('scaling', StandardScaler()), 
                        ('gaussian-nb', GaussianNB()) ]) 
        },
    'expt_16': { 
        'note': 'default Multinomial NB',
        'name': 'default Multinomial NB',
        'pl': Pipeline([ ('multi-nb', MultinomialNB()) ]) 
        },
    'expt_17': { 
        'note': 'scaled Multinomial NB',
        'name': 'scaled Multinomial NB',
        'pl': Pipeline([ ('scaling', StandardScaler()), 
                        ('multi-nb', MultinomialNB()) ]) 
        },
    'expt_18': { 
        'note': 'default LDA',
        'name': 'default LDA',
        'pl': Pipeline([ ('linear-da', LinearDiscriminantAnalysis()) ]) 
        },
    'expt_19': { 
        'note': 'scaled LDA',
        'name': 'scaled LDA',
        'pl': Pipeline([ ('scaling', StandardScaler()), 
                        ('linear-da', LinearDiscriminantAnalysis()) ]) 
        },
    'expt_20': { 
        'note': 'default QDA',
        'name': 'default QDA',
        'pl': Pipeline([ ('Quadratic-da', QuadraticDiscriminantAnalysis()) ]) 
        },
    'expt_21': { 
        'note': 'scaled QDA',
        'name': 'scaled QDA',
        'pl': Pipeline([ ('scaling', StandardScaler()), 
                        ('Quadratic-da', QuadraticDiscriminantAnalysis()) ]) 
        },
    'expt_22': { 
        'note': 'default (multi-class) Logistic regression',
        'name': 'default (multi-class) Logistic regression',
        'pl': Pipeline([ ('log-reg', LogisticRegression(n_jobs=-1)) ]) 
        },
    'expt_23': { 
        'note': 'scaled default (multi-class) Logistic regression',
        'name': 'scaled default (multi-class) Logistic regression',
        'pl': Pipeline([ ('scaling', StandardScaler()), 
                        ('log-reg', LogisticRegression(n_jobs=-1)) ]) 
        },
    # gridsearch cv the best performers from above ################################
    # - kNN
    'expt_24': { 
        'note': 'gridsearch cv on kNN',
        'name': 'gridsearch cv on kNN',
        'pl': GridSearchCV( Pipeline([ ('knn', KNeighborsClassifier(n_jobs=-1)) ]), 
                            param_grid=dict(knn__n_neighbors=[3,12,20]), 
                            n_jobs=-1 ) 
        },
    # - scaled rbf SVM
    'expt_25': { 
        'note': 'gridsearch cv on scaled rbf svm',
        'name': 'gridsearch cv on scaled rbf svm',
        'pl': GridSearchCV( Pipeline([ ('scaling', StandardScaler()), 
                                        ('rbf_svm', SVC(kernel='rbf', cache_size=1000)) ]),
                            param_grid=dict(rbf_svm__C=[0.1,1.0,10], 
                                            rbf_svm__gamma=[0.0001,0.01,0.1],
                                            rbf_svm__class_weight=[None, 'balanced']),
                            n_jobs=-1) 
        },
    # - scaled RF
    'expt_26': { 
        'note': 'gridsearch cv on scaled default RF',
        'name': 'gridsearch cv on scaled default RF',
        'pl': GridSearchCV( Pipeline([ ('scaling', StandardScaler()), 
                                        ('random_forest', RandomForestClassifier(n_jobs=-1)) ]), 
                            param_grid=dict(random_forest__n_estimators=[3,50,100],
                                            random_forest__max_features=[10,100,'auto']),
                            n_jobs=-1)
        },
    # narrower gridsearch on three models above #################################### 
    # - kNN
    'expt_27': { 
        'note': 'focused gridsearch cv on kNN',
        'name': 'focused gridsearch cv on kNN',
        'pl': GridSearchCV( Pipeline([ ('knn', KNeighborsClassifier(n_jobs=-1)) ]), 
                            param_grid=dict(knn__n_neighbors=range(2,12), 
                                            knn__weights=['distance','uniform']), 
                            n_jobs=-1 ) 
        },
    # - scaled rbf SVM      
    'expt_28': { 
        'note': 'focussed gridsearch cv on scaled rbf svm',
        'name': 'focussed gridsearch cv on scaled rbf svm',
        'pl': GridSearchCV( Pipeline([ ('scaling', StandardScaler()), 
                                        ('rbf_svm', SVC(kernel='rbf', cache_size=2000)) ]),
                            param_grid=dict(rbf_svm__C=[1,2,5,10], 
                                            rbf_svm__gamma=[0.001,0.005,0.01,'auto'],
                                            rbf_svm__class_weight=[None, 'balanced']),
                            n_jobs=-1) 
        },
    # - scaled RF
    'expt_29': { 
        'note': 'focussed gridsearch cv on scaled default RF',
        'name': 'focussed gridsearch cv on scaled default RF',
        'pl': GridSearchCV( Pipeline([ ('scaling', StandardScaler()), 
                                        ('random_forest', RandomForestClassifier(n_jobs=-1)) ]), 
                            param_grid=dict(random_forest__n_estimators=[10,100,500,1000],
                                            random_forest__max_features=[10,20,30,'auto']),
                            n_jobs=-1)
        },
    # best results of gridsearch'd models above #################################### 
    # - best kNN
    'expt_30': { 
        'note': 'best gridsearch result for kNN',
        'name': 'Neighborhood Treatment Plant Fence',
        'pl': Pipeline([ ('knn', KNeighborsClassifier(n_jobs=-1, 
                                                        weights='distance', 
                                                        n_neighbors=4)) ])
        },
    # - best scaled rbf SVM      
    'expt_31': { 
        'note': 'best gridsearch result for scaled rbf svm',
        'name': 'Small Popcorn Treatment Plant Fence',
        'pl': Pipeline([ ('scaling', StandardScaler()), 
                        ('rbf_svm', SVC(kernel='rbf', 
                                        cache_size=2000,
                                        C=10.0,
                                        gamma='auto',
                                        class_weight='balanced')) ])    
        },
    # - best scaled RF
    'expt_32': { 
        'note': 'best gridsearch result for scaled RF',
        'name': 'Small Wooded Treatment Plant Fence',
        'pl': Pipeline([ ('scaling', StandardScaler()), 
                        ('random_forest', RandomForestClassifier(n_jobs=-1,
                                                                    n_estimators=500,
                                                                    max_features='auto')) ])
        },
    # ensemble decision tree classifer that didn't get run earlier #################################### 
    'expt_33': { 
        'note': 'ExtraTrees',
        'name': 'ExtraTrees',
        'pl': Pipeline([ ('extra-trees', ExtraTreesClassifier(n_jobs=-1)) ]) 
        },
    'expt_34': { 
        'note': 'scaled default ExtraTrees',
        'name': 'scaled default ExtraTrees',
        'pl': Pipeline([ ('scaling', StandardScaler()), ('extra-trees', ExtraTreesClassifier(n_jobs=-1)) ]) 
        },
    # bagging versions of three best classifiers ##################################
    # - kNN
    'expt_35': { 
        'note': 'bagging on best gridsearched kNN estimator',
        'name': 'Sack of Flanders',
        'pl': BaggingClassifier( 
                    Pipeline([ ('knn', KNeighborsClassifier(n_jobs=-1, 
                                                            weights='distance', 
                                                            n_neighbors=4)) ]),
                    n_jobs=-1,
                    n_estimators=10)
                     
        },
    # - best scaled rbf SVM      
    'expt_36': { 
        'note': 'bagging on best gridsearch scaled rbf svm',
        'name': 'Sack of small popcorn',
        'pl': BaggingClassifier( 
                    Pipeline([ ('scaling', StandardScaler()), 
                            ('rbf_svm', SVC(kernel='rbf', 
                                            cache_size=2000,
                                            C=10.0,
                                            gamma='auto',
                                            class_weight='balanced')) ]),    
                    n_jobs=-1,
                    n_estimators=10)
        },
    # - best scaled RF
    'expt_37': { 
        'note': 'bagging on best gridsearch result for scaled RF',
        'name': 'Sack of small shrubs',
        'pl': BaggingClassifier( 
                    Pipeline([ ('scaling', StandardScaler()), 
                            ('random_forest', RandomForestClassifier(n_jobs=-1,
                                                                    n_estimators=500,
                                                                    max_features='auto')) ]),
                    n_jobs=-1,
                    n_estimators=10)
        },
    # adaboost with best RF (must supports class weights) #####################
    # - best scaled RF
    'expt_38': { 
        'note': 'adaboost on best gridsearch result for scaled RF',
        'name': 'On the shoulders of Ents',
        'pl':  Pipeline([ ('scaling', StandardScaler()), 
                            ('adaboost_random_forest', AdaBoostClassifier( 
                                                            RandomForestClassifier(n_jobs=-1,
                                                                                    n_estimators=500,
                                                                                    max_features='auto'),
                                                            n_estimators=100)) ])
        },
    # ensemble voting ################################################
    # - gridsearch voting w/ best three stand-alone models 
    'expt_39': { 
        'note': 'gs over voting across best gs models',
        'name': 'gs over voting across best gs models',
        'pl': GridSearchCV( 
                    VotingClassifier( estimators=[
                        ('gs_knn', Pipeline([ ('knn', KNeighborsClassifier(n_jobs=-1, 
                                                            weights='distance', 
                                                            n_neighbors=4)) ])),
                        ('gs_svm', Pipeline([ ('scaling', StandardScaler()), 
                                                ('rbf_svm', SVC(kernel='rbf', 
                                                                probability=True,
                                                                cache_size=2000,
                                                                C=10.0,
                                                                gamma='auto',
                                                                class_weight='balanced')) ])),    
                        ('gs_rf', Pipeline([ ('scaling', StandardScaler()), 
                                                ('random_forest', RandomForestClassifier(n_jobs=-1,
                                                                        n_estimators=500,
                                                                        max_features='auto')) ])) ]),
                    param_grid=dict(voting=['hard','soft']),
                    n_jobs=-1)
        },
    # - gridsearch voting w/ bagged combos 
    'expt_40': { 
        'note': 'gs over voting across bagged best gs models',
        'name': 'gs over voting across bagged best gs models',
        'pl': GridSearchCV( 
                    VotingClassifier( estimators=[
                        ('bag_knn', BaggingClassifier( 
                                        KNeighborsClassifier(n_jobs=-1, 
                                                            weights='distance', 
                                                            n_neighbors=4), 
                                        n_jobs=-1,
                                        n_estimators=10)),
                        ('bag_svm', BaggingClassifier( 
                                        Pipeline([ ('scaling', StandardScaler()), 
                                                    ('rbf_svm', SVC(kernel='rbf', 
                                                                    probability=True,
                                                                    cache_size=2000,
                                                                    C=10.0,
                                                                    gamma='auto',
                                                                    class_weight='balanced')) ]),    
                                        n_jobs=-1,
                                        n_estimators=10)),
                        ('bag_rf', BaggingClassifier( 
                                        Pipeline([ ('scaling', StandardScaler()), 
                                                    ('random_forest', RandomForestClassifier(n_jobs=-1,
                                                                                n_estimators=500,
                                                                                max_features='auto')) ]),
                                        n_jobs=-1,
                                        n_estimators=10))]),
                        param_grid=dict(voting=['hard','soft']),
                        n_jobs=-1)
        },
    # - gridsearch voting w/ bagged + boosted rf
    'expt_41': { 
        'note': 'gs over voting across bagged + boosted best gs models',
        'name': 'gs over voting across bagged + boosted best gs models',
        'pl': GridSearchCV( 
                    VotingClassifier( estimators=[
                        ('bag_knn', BaggingClassifier( 
                                        KNeighborsClassifier(n_jobs=-1, 
                                                            weights='distance', 
                                                            n_neighbors=4), 
                                        n_jobs=-1,
                                        n_estimators=10)),
                        ('bag_svm', BaggingClassifier( 
                                        Pipeline([ ('scaling', StandardScaler()), 
                                                    ('rbf_svm', SVC(kernel='rbf', 
                                                                    probability=True,
                                                                    cache_size=2000,
                                                                    C=10.0,
                                                                    gamma='auto',
                                                                    class_weight='balanced')) ]),    
                                        n_jobs=-1,
                                        n_estimators=10)),

                        ('boost_rf', Pipeline([ ('scaling', StandardScaler()), 
                                                ('adaboost_random_forest', AdaBoostClassifier( 
                                                                            RandomForestClassifier(n_jobs=-1,
                                                                                                n_estimators=500,
                                                                                                max_features='auto'),
                                                                            n_estimators=100)) ])) ]),


                        param_grid=dict(voting=['hard','soft']),
                        n_jobs=-1)
        },
    # - fix vote=soft for 39-40 (41?) & train on full data  ############################# 
    #   - (expt 39 w/o gs + soft vote)
    'expt_42': { 
        # "3-party system" trained this model on the original data 
        #'name': 'Basic three-party system',
        #'note': 'soft voting with best gs models',
        # "E Pluribus Unum" trained this model on the expanded data 
        'name': 'E pluribus unum',
        'note': 'soft voting with best gs models on expanded dataset',
        'pl': VotingClassifier( estimators=[
                        ('gs_knn', Pipeline([ ('knn', KNeighborsClassifier(n_jobs=-1, 
                                                            weights='distance', 
                                                            n_neighbors=4)) ])),
                        ('gs_svm', Pipeline([ ('scaling', StandardScaler()), 
                                                ('rbf_svm', SVC(kernel='rbf', 
                                                                probability=True,
                                                                cache_size=2000,
                                                                C=10.0,
                                                                gamma='auto',
                                                                class_weight='balanced')) ])),    
                        ('gs_rf', Pipeline([ ('scaling', StandardScaler()), 
                                                ('random_forest', RandomForestClassifier(n_jobs=-1,
                                                                                        n_estimators=500,
                                                                                        max_features='auto')) ])) ],
                    voting='soft')

        },
    #   - (expt 40 w/o gs + soft vote)
    'expt_43': { 
        'note': 'soft voting with bagged gs models',
        'name': 'PACs and the three-party system',
        'pl': VotingClassifier( estimators=[
                        ('bag_knn', BaggingClassifier( 
                                        KNeighborsClassifier(n_jobs=-1, 
                                                            weights='distance', 
                                                            n_neighbors=4), 
                                        n_jobs=-1,
                                        n_estimators=10)),
                        ('bag_svm', BaggingClassifier( 
                                        Pipeline([ ('scaling', StandardScaler()), 
                                                    ('rbf_svm', SVC(kernel='rbf', 
                                                                    probability=True,
                                                                    cache_size=2000,
                                                                    C=10.0,
                                                                    gamma='auto',
                                                                    class_weight='balanced')) ]),    
                                        n_jobs=-1,
                                        n_estimators=10)),
                        ('bag_rf', BaggingClassifier( 
                                        Pipeline([ ('scaling', StandardScaler()), 
                                                    ('random_forest', RandomForestClassifier(n_jobs=-1,
                                                                                n_estimators=500,
                                                                                max_features='auto')) ]),
                                        n_jobs=-1,
                                        n_estimators=10))],
                    voting='soft')
        },
    #   - (expt 41 w/o gs + soft vote)
    'expt_44': { 
        'note': 'voting classifier: 2x bags + boosted RF w/ soft voting',
        'name': 'SuperPACs ruin everything',
        'pl': VotingClassifier( estimators=[
                        ('bag_knn', BaggingClassifier( 
                                        KNeighborsClassifier(n_jobs=-1, 
                                                            weights='distance', 
                                                            n_neighbors=4), 
                                        n_jobs=-1,
                                        n_estimators=10)),
                        ('bag_svm', BaggingClassifier( 
                                        Pipeline([ ('scaling', StandardScaler()), 
                                                    ('rbf_svm', SVC(kernel='rbf', 
                                                                    probability=True,
                                                                    cache_size=2000,
                                                                    C=10.0,
                                                                    gamma='auto',
                                                                    class_weight='balanced')) ]),    
                                        n_jobs=-1,
                                        n_estimators=10)),
                        ('boost_rf', Pipeline([ ('scaling', StandardScaler()), 
                                                ('adaboost_random_forest', AdaBoostClassifier( 
                                                                                RandomForestClassifier(n_jobs=-1,
                                                                                                n_estimators=500,
                                                                                                max_features='auto'),
                                                                                n_estimators=100)) ])) ],
                    voting='soft')
        },
    # Include inferred class distributions in best stand-alone models of SVM, RF ################## 
    'expt_45': { 
        'note': 'add class weights to expt_32',
        'name': 'Yeah I work out',
        'pl': Pipeline([ ('scaling', StandardScaler()), 
                        ('random_forest', RandomForestClassifier(n_jobs=-1,
                                                                    n_estimators=500,
                                                                    max_features='auto',
                                                                    class_weight = {0:0.098, 
                                                                                    1:0.111, 
                                                                                    2:0.104, 
                                                                                    3:0.102, 
                                                                                    4:0.098, 
                                                                                    5:0.088, 
                                                                                    6:0.095, 
                                                                                    7:0.103, 
                                                                                    8:0.098, 
                                                                                    9:0.102})) ])
        },
    'expt_46': { 
        'note': 'add class weights to expt_36',
        'name': 'Oh you work out?',
        'pl': BaggingClassifier( 
                    Pipeline([ ('scaling', StandardScaler()), 
                            ('rbf_svm', SVC(kernel='rbf', 
                                            cache_size=2000,
                                            C=10.0,
                                            gamma='auto',
                                            class_weight = {0:0.098, 
                                                            1:0.111, 
                                                            2:0.104, 
                                                            3:0.102, 
                                                            4:0.098, 
                                                            5:0.088, 
                                                            6:0.095, 
                                                            7:0.103, 
                                                            8:0.098, 
                                                            9:0.102})) ]),    
                    n_jobs=-1,
                    n_estimators=10)
        },
    # 
    # As of the time of writing, using the MLPClassifier requires building the   
    #   developer branch of sklearn. If you want to use these experiments, 
    #   the sklearn docs include a ref for building this version:
    #   http://scikit-learn.org/stable/developers/contributing.html#git-repo
    #   Then, you can uncomment the next few experiments below (+ 52) to run them. 
    #
    # neural network experiments ################################################
    # - sklearn's MLPClassifier
#    'expt_47': { 
#        'note': 'gridsearch multilayer perceptron, using tips from dev docs',
#        'name': 'tbd',
#        'pl': GridSearchCV( 
#                    Pipeline([ ('scaling', StandardScaler()), 
#                                ('mlp', MLPClassifier()) ]), 
#                    param_grid=dict( mlp__alpha=10.0**-np.arange(1, 7),
#                                    mlp__hidden_layer_sizes=[(50, ), (100, ), (200, )],
#                                    mlp__activation=['logistic', 'tanh', 'relu'],
#                                    mlp__algorithm=['l-bfgs', 'sgd', 'adam']),
#                    n_jobs=-1)
#        },
#    # - v2 of sklearn's MLPClassifier
#    'expt_48': { 
#        'note': 'v2 of gridsearch multilayer perceptron, modifying param_grid',
#        'name': 'tbd',
#        'pl': GridSearchCV( 
#                    Pipeline([ ('scaling', StandardScaler()), 
#                                ('mlp', MLPClassifier(activation='relu')) ]), 
#                    param_grid=dict( mlp__alpha=10.0**-np.arange(-1,6),
#                                    mlp__hidden_layer_sizes=[(50,), 
#                                                            (100,), 
#                                                            (200,), 
#                                                            (50,50), 
#                                                            (100,100), 
#                                                            (200,200), 
#                                                            (50,50,50), 
#                                                            (100,100,100), 
#                                                            (200,200,200)],
#                                    mlp__algorithm=['l-bfgs', 'adam']),
#                    n_jobs=-1)
#        },
#    # - gridsearch wide MLP hidden layers
#    'expt_49': { 
#        'note': 'v3 of gridsearch multilayer perceptron, modifying param_grid',
#        'name': 'tbd',
#        'pl': GridSearchCV( 
#                    Pipeline([ ('scaling', StandardScaler()), 
#                                ('mlp', MLPClassifier(activation='relu', verbose=True)) ]), 
#                    param_grid=dict( mlp__alpha=10.0**-np.arange(-2,5),
#                                    mlp__hidden_layer_sizes=[(200,), 
#                                                            (500,), 
#                                                            (1000,), 
#                                                            (200,200), 
#                                                            (500,500), 
#                                                            (1000,1000), 
#                                                            (500,500,500)],
#                                    mlp__algorithm=['l-bfgs', 'adam']),
#                    n_jobs=-1)
#        },
    # revisit SVM with poly kernel gridsearch ##################################################
    'expt_50': { 
        'note': 'gridsearch poly kernel degree with scaled svm',
        'name': 'gridsearch poly kernel degree with scaled svm',
        'pl': GridSearchCV( Pipeline([ ('scaling', StandardScaler()), 
                                        ('svm', SVC(cache_size=2000,
                                                    kernel='poly', 
                                                    gamma='auto')) ]),
                            param_grid=dict(svm__C=[0.1, 0.5, 1.0, 5.0, 10.0, 15.0],
                                            svm__degree=np.arange(2,12)),
                            n_jobs=-1) 
        },
#    # dimensionality reduction + kNN ######################################################
#    # note: this doesn't work because TSNE doesn't implement a transform method. Pipeline throws 
#    #          an error on import about this, so leave this commented out.
#    'expt_51': { 
#        'note': 'gridsearch over tSNE dim reduction + kNN',
#        'name': 'gridsearch over tSNE dim reduction + kNN',
#        'pl': GridSearchCV( Pipeline([ 
#                                ('tsne', TSNE(verbose=1)), 
#                                ('knn', KNeighborsClassifier(n_jobs=-1)) ]), 
#                            param_grid=dict(tsne__n_components=[2,3,4],
#                                            tsne__perplexity=[20,30,40,50],
#                                            tsne__learning_rate=[400,700,1000],
#                                            knn__n_neighbors=range(2,10), 
#                                            knn__weights=['distance','uniform']), 
#                            n_jobs=-1 ) 
#        },
    # best MLP from gridsearch (note: out of order due to run time!) ######################### 
    # {'mlp__hidden_layer_sizes': (1000, 1000), 'mlp__algorithm': 'l-bfgs', 'mlp__alpha': 10.0} 
#    'expt_52': { 
#        'note': 'best MLP from gridsearch',
#        'name': 'Pinky and the Brain',
#        'pl': Pipeline([ ('scaling', StandardScaler()), 
#                        ('mlp', MLPClassifier(activation='relu', 
#                                                hidden_layer_sizes=(1000,1000), 
#                                                algorithm='l-bfgs', 
#                                                alpha=10.0, 
#                                                verbose=True)) ]) 
#        },

    } # end of experiment_dict


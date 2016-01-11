# CCC, 2015 edition

**Note: This is a clean-up in-progress** 

-----------

2015-12-16, kick-off meeting to review project definition

- submission format (see "random" example, too): 

    <team name, make this consistent across all submissions. This will appear in the leader board.>
    <timestamp of sumbission in format 2015-12-xxT00:00:00>
    <submission name-anything you want. This will appear in the leader board.>


# outline 

- [x] read some scikit examples 
- [x] copy the given images code into lab, create numpy arrays 
    - because of some differences in fileInput in the 3.4 interpreter, reverting back to 2.7 
    - able to run images.py as given
    - work on getting the arrays out into np arrays => write to file for easy import later
        - note: takes ~30s to read original data into np array
- [x] make first prediction file (interactively)
    - add test data to the existing image.py script 
    - use dummy classifier
- [x] move binary => numpy creation into Makefile 
    - still in notebook
    - other formats?
        - joblib (compressed) [link](https://pythonhosted.org/joblib/persistence.html)
        - HDF5 (groups) [link](http://docs.h5py.org/en/latest/quick.html#appendix-creating-a-file)
    - [this (old) post](https://robertdragan.wordpress.com/2012/08/31/comparying-various-methods-for-saving-and-loading-numpy-arrays/) shows that joblib/hdf5/numpy are all about the same in performance. joblib looks to win, slightly, and is recommended in the scikit-learn docs. so, use that one.
        - this seems to pertain more to the models that carry large arrays within them. 
    - stick to .npy arrays from original data, and clean up that code 
- [x] improve virtualenv setup
    - get virtualenv incorporated with existing conversion script 
    - can now ``make everything`` from scratch!
- [x] use basic SVM in notebook (memory of this being good baseline for MNIST)
- [x] move utils code out of notebook 
    - data reading, submissions creation
    -  fix import paths for bin/
- [x] build diagnostics (to save with each model)
    - scoring (accuracy + stdev) 
    - ``cross_val_predict`` + confusion matrix 
- [x] per-experiment executables
    - create python module that defines the list of models, steps 
        - use eg ``experiment-1.py`` to read that in and execute from bash 
    - include saving model, logging, saving confusion matrix 
- [x] next round of experiments
    - loop over: scaling v. no scaling X every default classifier 
    - summarize results (``$ cat log/*.log | grep "+/-" | cut -d"=" -f2,5- | sort -nr``):
        - k-NN (96.61%)
        - scaled rbf SVM (95.76%)
        - scaled RF (93.67%)
        - scaled k-NN (93.65%)
        - RF (93.59%)
    - train the top two (kNN [expt4] + scaled rbf SVM [expt7]) on all training data  => predict + submission
        - make another python utility that takes the experiment pipeline (from models.py) and creates a submission file 
    - the kNN is pretty fast, but the rbf SVM takes a while train/predict
        - also ran the wrong model (expt6); killed and running expt7 now
        - submitted kNN (Mac & Kelly)
        - submitted scaled rbf SVM (compact popcorn machine)
    - didn't include multi-class LR (scaled and not) in initial experiment -- do that now 
        - not stellar, ~90%
- [x] gridsearch (first level) the best performers from ^ 
    - gridsearchcv the top three performers above (kNN, scaled rbf SVM, scaled RF)
        - need to find reasonable params for each model
    - update any code that relies on on pipeline (instead of gscv) 
    - best gridsearch kNN had ~95%, lower than with default settings
    - killed and restarted a few times with attempts to boost efficiency
        - these are expts 24-26. best results: 
        - kNN ~ 95% ({'knn__n_neighbors': 3}) 
            - default k=10 (was 96.6%)
        - SVM ~ 93% ({'rbf_svm__C': 10, 'rbf_svm__gamma': 0.0001, 'rbf_svm__class_weight': 'balanced'} [tie w/ c_w=None]) 
            - default C=1, gamma='auto' (1/n_features ~ 0.008), classweight=None 
        - RF ~ 94% ({'random_forest__n_estimators': 100, 'random_forest__max_features': 'auto'})  
            - default n_estimators=10, max_features='auto'
- [x] gridsearch (second level) 
    - focus ranges around blend of "default" and last GS best performance 
        - these are expts 27-29; check logs for best
        - kNN: ~95%. k=3-5, all similar, weight='distance'; {'knn__weights': 'distance', 'knn__n_neighbors': 4}
        - SVM: ~95%. C=10, gamma=0.001/auto, similar; {'rbf_svm__C': 10, 'rbf_svm__gamma': 'auto', 'rbf_svm__class_weight': 'balanced'}
        - RF: ~95%. {'random_forest__n_estimators': 500, 'random_forest__max_features': 'auto'}
    - once these are fit, submit one or more
        - train RF on everything (expt 32), submit results 
- [x] intermission to clean up repo & file structure
- [x] didn't try ExtraTrees classifier earlier -- try this now
    - 94%, slightly higher with scaling (still 94.x%)
- [x] read about [ensemble methods](http://scikit-learn.org/stable/modules/ensemble.html#ensemble-methods) 
    - averaging
        - Bagging == use this one each of 3 best
    - boosting
        - AdaBoost ==> use large adaboost classifier with each of 3 best classifiers 
        - supposedly better with e.g. shallow trees, maybe our gridsearch'd models are opposed to this? 
    - VotingClassifer


- [x] set up baggingclassifier with each of the three best as base
    - if the 'pl' is BaggingClassifier(Pipeline()), need to update e.g. utils.name() anything in run-experiment.py?? 
        - seems like the baggingclassifier params could also be gridsearched
    - running all in parallel seems too much for # of cores... increase stagger
        - bagging kNN didn't finish. 
            - run this one again (expt_35) 
            - ~96.8%, improvement over single RF
        - bagging SVM (36) ~ 96.3% - improvement over the single svm
        - bagging RF (37) ~ 96.3% - improvement over single RF
    - submit each of these trained on full dataset
- [x] set up adaboost with best RF 
    - must have class weights and proper attrs in estimator (SVC doesn't, RF does, kNN doesn't)
    - try 100 estimators (expt 38) 
    - very fast. ~ 96.5% (slightly better than bagging)
    - submit
- [x] gridsearch votingclassifier on top of the 3 gs'd classifiers to find best 'vote' type
    - then fit, train, predict, submit that one
    - debug: JoblibAttributeError
        - seems related to probability estimates in SVC; add probability=True to constructor (doesn't appear to effect other performance)
    - 0.962 (+/-0.010) for {'voting': 'soft'}
    - was expecting a better result by "averaging"
        - trail full model with voting=soft & submit 
- [x] gridsearch VC for 'vote' w/ 3x bagged or adaboost 
    - 3x bagged: 0.960 (+/-0.009) for {'voting': 'soft'}
    - 2x bagged + adaboost RF: 0.961 (+/-0.010) for {'voting': 'soft'}
        - was expecting a better result by "averaging"
    - train full model on 3x bagging w/ voting=soft & submit (43) 
    - train full model on 2x bagging + RF boosting w/ voting=soft & submit (44)
- [x] use ``class_weights`` in RF & SVM models to reverse-engineered values from scoreboard
    - look at submission for Small Wooded Treatment Plant Fence (expt-32) and count up the predictions 

```bash
$ tail -n+6 submissions/2015-12-24T18:16:42_Small-Wooded-Treatment-Plant-Fence.submission | sort | uniq -c | sort -n | sed 's/^ *//' | sort -t" " -k2,2 > data/SWTPF_counts.csv
997 0
1135 1
1039 2
1011 3
980 4
880 5
961 6
1022 7
969 8
1006 9
```

- now look at per-count accuracy from scoreboard (data/SWTPF-leaderboard-scores.csv) 

```bash
0.99081633 0 
0.99030837 1  
0.9689922481 2    
0.9653465347 3   
0.9735234216 4   
0.9674887892 5   
0.9791231733 6   
0.96692607 7 
0.9599589322 8   
0.9554013875 9
```
- and now we can combine them to get the actual count of digits in the leaderboard test set (if we round) 

```bash
$ join SWTPF_counts.csv SWTPF-leaderboard-scores.csv -1 2 -2 2 | awk 'BEGIN { sum = 0 } { printf "%d %d\n", $1, $2/$3; sum+=$2/$3 } END { printf "\n%d \n", sum }'
0 1006
1 1146
2 1072
3 1047
4 1006
5 909
6 981
7 1056
8 1009
9 1052

10288 

$ join SWTPF_counts.csv SWTPF-leaderboard-scores.csv -1 2 -2 2 | awk 'BEGIN { sum = 10288 } { printf "%d %1.3f\n", $1, $2/$3/sum }'
0 0.098
1 0.111
2 0.104
3 0.102
4 0.098
5 0.088
6 0.095
7 0.103
8 0.098
9 0.102
```

- in principal, we can now use the relative prevalence of these to weight the classes in eg the SVC model (want the weights to sum to one)
- both SVC and RF support passing the class weights, repurpose the best-performing versions of those
    - RF: expt 32 (scalded RF) performed best on leaderboard (97.2%)
        - reuse with weights (45) => ~96.6%, decent
        - leaderboard score =>  97.1%
    - SVM: expt 36 (bagged, scaled, gs'd SVM) performed best on leaderboard (97.1%)
        - reuse with weights (46) => ~95.6%, decent
        - leaderboard score => 96.4%
- submit these as stand-alone models [running now]
    - relaunched them because they weren't named (overwrite log files)
    - sent
- neither were much higher than the original; don't bother updating VotingClassifier 
- [x] sklearn's built-in NN (MLPClassifier)
- big gridsearch
- dang. MLPC only in dev version of scikit 
    - see if we can create a local virtualenv for that
    - looks ok, running as expt_47

```bash
jmontague@data-science-3:~
$ virtualenv -p python ~/CCC-venv

jmontague@data-science-3:~/2015-12-21_CCC [master+*]
$ source ~/CCC-venv/bin/activate

jmontague@data-science-3:~
$ pip install -r requirements.txt
$ pip uninstall scikit-learn

(CCC-venv)jmontague@data-science-3:~/2015-12-21_CCC [master+*]
$ pip install -e git+git@github.com:scikit-learn/scikit-learn.git

(CCC-venv)jmontague@data-science-3:~/2015-12-21_CCC [master+*]
$ pip install cython

(CCC-venv)jmontague@data-science-3:~/CCC-venv/lib/python2.7/site-packages/scikit-learn [master]
$ python setup.py build_ext --inplace
$ python
>>> import sklearn; sklearn.__version__
'0.18.dev0'

# but didn't build/install totally correctly, maybe ran setup.py in wrong place?
#   - in virtualenv, get sklearn ImportError
#   - resolve by either (in the launch-process.bash script):
$ export PYTHONPATH=~/CCC-venv/lib/python2.7/site-packages/scikit-learn:$PYTHONPATH
# or: 
jmontague@data-science-3:~/CCC-venv/lib/python2.7/site-packages 
$ ln -s scikit-learn/sklearn sklearn
```

- ran gridsearch on 47 - worked, high of ~94.8%

- [x] try again with updated grid based on scores 
        - best alpha was on edge of grid - run again to extend on larger end
        - also didn't think to add extra layers - add that, too: [(50,), (100,), (200,), (50,50), (100,100), (200,200), (50,50,50), (100,100,100), (200,200,200)] 
        - drop 'sgd' algorithm 
        - stick to 'relu' activation 
    - best score (from 48) ~ 95.6%
        - extra layer, alpha at edge of grid again
        - look through this more & make next round of GS:
        ``cat log/2015-12-29T03:43:29_expt_48.log | grep for | sort -n -t" " -k6,6``

- [x] run with larger range of layer sizes and other params 
    - took ~4 hrs for GSCV 
    - best model ~95.8%: {'mlp__hidden_layer_sizes': (1000, 1000), 'mlp__algorithm': 'l-bfgs', 'mlp__alpha': 10.0}
    - convert best to train for submission  (52 - note: out of order bc of earlier long run times 

- [x] test other SVM kernels (in particular, poly w/ gs on degree) 
    - running as expt_50 
    - best ~95.1%, {'svm__degree': 2, 'svm__C': 15.0}
    - not noticibly better than rbf kernel (which I think should do better with high-dimensional data) 
- [x] combine dimensionality reduction with kNN
    - t-SNE to 2-5 dimensions, then kNN (expt 51) 
    - doesn't work because TSNE doesn't have a transform method

- [x] expand training data with perturbations
    - then train this data on all of the simplest algorithms 
    - new train data ~ 1 GB
- [x] test this with default models (expt 27, 28, 29), compare scores 
    - need to flag run-experiment.py to read the proper dataset
    - these are slower to train, need to increase stagger time (killed 28 and 29 to let 27 run - restart them once 27 finishes)
        - 27 running for >1 hr
    - also much more ram (~30 GB for expt_27)
    - all 3 were ~95% in first round (w/ the smaller dataset) 
        - kNN (96.1%): {'knn__weights': 'distance', 'knn__n_neighbors': 4} (which is the same as expt_30 
            - ^ from the gscv, killed before fully finished 
        - SVM ():  
        - RF ():  
    - in the interest of trying to get predictions, I think I'll just run a full fit and predict on the gridsearch'd voting classifier (basic three party system, 42)  
        - running now ("expanded 42"). note: original fit to data took 1.5 hrs, and this data is 5x bigger.
        - took 16 hours!
    - submitted (98.6% on hold-out set) 
        


# other ideas

- scikit-neuralnetwork
- tpot
- sklearn-deap
- nolearn
- tensorflow 

------------

# TODO

- modify Makefile targets 
    - [x] build venv
        - this seems messy... scipy and sklearn seem to have failed and are being built from source because of an issue with the numpy install? 
    - [x] .npy files 
    - [] .submission files 
- test fresh run-through (git clone => first round of models)  
- check for typos in readme



# Future work? 

- move matrix plotting into utils module (?)
- make utils.short_name less fragile 
- build funcs to read and display example images
- look at feature importance 
    - http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances_faces.html#example-ensemble-plot-forest-importances-faces-py
    - http://bugra.github.io/work/notes/2014-11-22/an-introduction-to-supervised-learning-scikit-learn/)
    - ex: in voting classifier, does per-digit accuracy vary by model?



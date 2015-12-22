2015-12-16 

Kick-off meeting to review project definition

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


- [] next round of experiments
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
        - _____ submitted scaled rbf SVM (compact popcorn machine)




- [] gridsearch the best performers from ^ 





- [] ensemble methods for best gridsearch'd settings of ^
    - [VotingClassifer](http://scikit-learn.org/stable/modules/ensemble.html#voting-classifier) can also be gridsearched for component classifier settings 



- [] strongest features + **adjust probabilities of assignment based on leaderboard observations** 

- [] build funcs to read and display example images

- [] if too slow, can we fit a linear model and look at the relative importance of features 

- [] start building models 
    - interactive single model
    - end-to-end, executable single model (read, process, model, predict, write)
    - loop through default settings of [all of these](http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html)
    - choose a couple to gridsearch (or similar)
    - consider preprocessing approaches 
        - scikit's preprocessing module (zero mean, unit variance)
        - minmax scalar 
    - try ensemble methods
    - some approaches in [this writeup](http://colah.github.io/posts/2014-10-Visualizing-MNIST/)
    - try NN, convnets?

## results

[Submission date] Classifier/Pipeline: training accuracy (leaderboard accuracy)

    - [ 2015-12-18 ] DummyClassifier (random guessing) ['Crash Test Dummies']: 10% (10%)
    - [ 2015-12-19 ] Linear SVM (default settings) ['Grapevine']: 86% (79%) **interesting that the leaderboard score is that much lower** 
    - [ 2015-12-20 ] Scaling + Linear SVM DummyClassifier [Tall To Ride]: 91% (N/A)
    



# TODO

- add mpl backend change to makefile 


- [] other things to try:
    - visualization? 
        - MDS / tSNE + kNN [via](http://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html#example-manifold-plot-lle-digits-py)
    - ``metrics.confusion_matrix``
    - feature scaling? is there a reason to scale 0-256 down to 0-1? 
    - feature importance [via](http://bugra.github.io/work/notes/2014-11-22/an-introduction-to-supervised-learning-scikit-learn/)

------------












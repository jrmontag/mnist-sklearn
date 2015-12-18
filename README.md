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


- [] improve virtualenv setup



- [] take model + pipeline out of notebook

- [] build funcs to read and display example images

- [] move from ipynb => executables asap 

- [] start building models 
    - interactive single model
    - end-to-end, executable single model (read, process, model, predict, write)
    - consider preprocessing approaches
    - loop through default settings of [all of these](http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html)
    - choose a couple to gridsearch (or similar)
    - try ensemble methods
    - some approaches in [this writeup](http://colah.github.io/posts/2014-10-Visualizing-MNIST/)
    - try NN, convnets?

- [] other things to try:
    - MDS / tSNE + kNN [via](http://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html#example-manifold-plot-lle-digits-py)
    - ``metrics.confusion_matrix``

------------












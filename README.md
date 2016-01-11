# MNIST + ``scikit-learn`` = :star2: 


This code was developed for a intra-team Kaggle-like modeling competition on the canonical [MNIST handwritten digits dataset](https://en.wikipedia.org/wiki/MNIST_database).  

We were given the training images and labels, the test images, and a simple Python script that read (and "displayed") the binary image data. We had two weeks to submit any number of prediction files for the test images, one prediction per line.  

The other rules were pretty simple: 

- write code in whatever language you want
- use whatever platform you want (laptop, EC2, tea leaves, stone tablet)  
- be prepared to present what you did to the team (in particular, this meant if you cheated and just downloaded the test labels, you were in for A Bad Time) 

Given the small size of the data and how much I :heart: ``scikit-learn``, I chose to use Python on a single server (a combination of MacBook and AWS EC2). All of the data processing and modeling code here is written in Python, though there are a couple of additional bash scripts to facilitate various pieces of the workflow. 

I think most of this code should work out of the box with Python 2.7 on OS X and Ubuntu (an isolated environment is recommended - the requirements.txt I used is included). In fact, if all goes as planned , the following commands will get you up and running with the data (original version ~10 MB, and ``.npy`` version ~400 MB), a shiny new virtual environment, and a handful of stored models with their prediction accuracies in confusion matrices (the expanded data set used to obtain the top score is obtained by a separate ``make`` command, and is ~2 GB):

```bash
$ git clone <this repo>
$ cd mnist-sklearn
$ make demo 
```

While I did test this a few different ways, I can't guarantee it'll work seamlessly! Feel free to poke around in the ``Makefile`` to see what is intended. For more words about the context and workflow, you can read [my longer post](http://www.joshmontague.com/). 


-----

## Notes

- I'd recommend using an isolated environment; I used ``virtualenv`` and the Makefile will you set that up 
    - for reasons unclear to me, when ``pip install``ing these requirements on Ubuntu, ``pip`` threw a ``Failed building wheel`` error for ``numpy, scipy,`` and ``scikit-learn``. It appeared to then recover and be happy by building them via ``setup.py``.  ``¯\_(ツ)_/¯`` 
- As of the time of writing, the development branch of ``sklearn`` is required to use the ``MLPClassifer`` (Multi-layer perceptron). See the additional notes below for instructions on setting this up. 
- This code was developed on both OS X and Ubuntu, which can lead to inconsistencies in the behavior of library behavior. At least once, I had to modify the ``matplotlib`` "backend" in the corresponding Python environment ``matplotlibrc`` file. In the end, I was using ``macosx`` and ``agg`` (on OS X and Ubuntu, respectively). 


### Installing the development branch of ``sklearn`` 

I sort of figured this out by trial an error (on Ubuntu, didn't test on OS X), so there may be a better way to do it. Nevertheless, here's what I did that worked:

- If you want to build a separate virtualenv from the one with stable ``sklearn`` (this is, after all part of the point of virtualenv), do the following:

```bash
# install similar libraries as before, plus cython
$ make skl-dev-env
# install sklearn from the git commit I used  
$ source dev-tmp-venv/bin/activate
$ pip install git+https://github.com/scikit-learn/scikit-learn.git@7cfa55452609c717c96b4c267466c80cc4038845
```

- Or, if you want to use the virtualenv that you've already built, you can:

```bash
# replace sklearn + install cython 
$ source dev-tmp-venv/bin/activate
$ pip uninstall sklearn
$ pip install cython
$ pip install git+https://github.com/scikit-learn/scikit-learn.git@7cfa55452609c717c96b4c267466c80cc4038845
``` 

 


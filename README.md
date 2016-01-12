# MNIST + ``scikit-learn`` = :star2: 

This code was developed for a intra-team Kaggle-like modeling competition on the canonical [MNIST handwritten digits dataset](https://en.wikipedia.org/wiki/MNIST_database).  

We were given the training images and labels, the test images, and a simple Python script that read (and "displayed") the binary image data. We had two weeks to submit any number of prediction files for the test images, one prediction per line. My highest accuracy model ([#42](https://github.com/jrmontag/mnist-sklearn/blob/master/models.py#L419)) scored 98.18% with no feature engineering. The same model with a minor amount of feature engineering ("added samples" [via image translation](https://github.com/jrmontag/mnist-sklearn/blob/master/expand-np-arrays.py)) scored a 98.68% and won the competition. 

The rules were pretty simple: 

- write code in whatever language you want
- use whatever platform you want (laptop, EC2, tea leaves, stone tablet)  
- be prepared to present what you did to the team (in particular, this meant if you cheated and just downloaded the test labels, you were in for A Bad Time) 

Given the small size of the data and how much I :heart: ``scikit-learn``, I chose to use Python on a single server (a combination of a MacBook and AWS EC2). All of the data processing and modeling code here is written in Python, though there are a couple of additional bash scripts to facilitate various pieces of the workflow. I think most of this code should work out of the box with Python 2.7 on OS X and Ubuntu (an isolated environment is recommended - the requirements.txt I used is included). 

**BUT WAIT!** I can do you one better: I included the entire setup that I used within this repo. If all goes as intended, the following commands will get you up and running\*. 

```bash
$ git clone <this repo>
$ cd mnist-sklearn
$ make demo 
$ open log/*.pdf
```

\*the prerequisites for using this code are having Python 2.7 (yeah, I know it's old), ``make``, and ``virtualenv`` installed. Your machine likely has ``make`` installed already. If needed, you can ``sudo pip install virtualenv``. 

## What's happening here?

The ``make demo`` command will do the following things:

- use ``virtualenv`` to create an isolated Python environment in this directory (and install all the necessary libraries) 
- download the raw binary data from [Yann LeCun's website](http://yann.lecun.com/exdb/mnist/) (about 10 MB) 
- convert those binary files to ``numpy`` arrays and write them to disk (about 400 MB) 
- start a set of sample model runs in the background 

The one-time environment setup may take a few minutes. The three sample models are staggered by the ``bash`` script and should be done in about 45 seconds. You can look at the data in ``log/`` to see what's going on, and look in ``saved_models/`` to see both the serialized models and per-model confusion matrices. 

## What should I do next?

For more standard usage, the designed approach is to add new Pipelines to ``models.py`` (with accompanying descriptions and names, used in file-nameing conventions). Then, update the ``SEQUENCE`` variable in ``launch-processes.bash`` - either using a single value, or a range via ``seq``. Each "experiment" (as the ``expt_*`` convention was intended), will create a new log file and all logs from a single use of ``launch-process.bash`` will share a timestamp for ease of separating your trials. 

Since some of the models can take minutes to hours to run, the recommended syntax is something like:

```bash
$ nohup bash launch-processes.bash > log/2016-01-12_expt-4-12.nohup.log &  
``` 

This will let you disconnect from the session while things are still running, and also log (in the nohup log) any unexpected exceptions that crash your code.  

Given the relatively small size of data, most of these models seem to be CPU bound. For optimal iteration time (and fun of watching ``htop``), a high-CPU-count server is the best approach. Go ask AWS for something from the C3 or C4 family of EC2 instances. 

As you dig in further, review the Notes section below for some important details.

Questions? [Let me know](https://www.twitter.com/jrmontag)! Otherwise, have fun classifying! 

-----

## Notes

- I'd recommend using an isolated environment; I used ``virtualenv`` and the Makefile will you set that up 
    - for reasons unclear to me, when ``pip install``ing these requirements on Ubuntu, ``pip`` threw a ``Failed building wheel`` error for ``numpy, scipy,`` and ``scikit-learn``. It appeared to then recover and be happy by building them via ``setup.py``.  ``¯\_(ツ)_/¯`` 
- As of the time of writing, the development branch of ``sklearn`` is required to use the ``MLPClassifer`` (Multi-layer perceptron). See the additional notes below for instructions on setting this up. 
- This code was developed on both OS X and Ubuntu, which can lead to inconsistencies in the behavior of library behavior. At least once, I had to modify the ``matplotlib`` "backend" in the corresponding Python environment ``matplotlibrc`` file. In the end, I was using ``macosx`` and ``agg`` (on OS X and Ubuntu, respectively). 
- The expanded data set (small, linear translations of the original data) used to obtain the top score is obtained by a separate ``make`` command, and is ~2 GB):
- As the get more complicated, models can take from order seconds (default k-nearest neighbors) to order hours (Multi-layer Perceptron). If using shared resources (or, anecdotally, to increase the efficiency of running multiple processes), I recommend increasing the ``SLEEPTIME`` variable in the launch script to something like 1-5 minutes. 


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

 


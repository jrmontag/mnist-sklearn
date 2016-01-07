# MNIST + ``scikit-learn`` = :star2: 


This code was developed for a intra-team Kaggle-like modeling competition on the canonical [MNIST handwritten digits dataset](https://en.wikipedia.org/wiki/MNIST_database).  

We were given the training images and labels, the test images, and a simple Python script that read (and "displayed") the binary image data. We had two weeks to submit any number of prediction files for the test images, one prediction per line.  

The other rules were pretty simple: 

- write code in whatever language you want
- use whatever platform you want (laptop, EC2, tea leaves, stone tablet)  
- be prepared to present what you did to the team (in particular, this meant if you cheated and just downloaded the test labels, you were in for A Bad Time) 

Given the size of the data and how much I :heart: ``scikit-learn``, I chose to use Python on a single server (combination of MacBook and EC2). All of the data processing and modeling code here is written in Python, and there are a couple of additional bash scripts to facilitate various pieces of the workflow. 

I think most of this code should work out of the box with Python 2.7 on OS X and Ubuntu (an isolated environment is recommended - the requirements.txt I used is included). However, installing the development version of ``sklearn`` is not out-of-the-box. See more notes below. 

```bash
[placeholder for basic instructions]
```

Ideally, you'll just ``git clone; cd; make`` and get quite a ways.

For more words about the context and workflow, you can read [my longer post](http://joshmontague.com). 

-----

## Notes

- I'd suggest using an isolated environment; I used ``virtualenv`` and the Makefile will you set that up 
- As of the time of writing, the development branch of ``sklearn`` is required to use the ``MLPClassifer`` (Multi-layer perceptron). See the additional notes below for instructions on setting this up. 
- This code was developed on both OS X and Ubuntu, which can lead to inconsistencies in the behavior of library behavior. At least once, I had to modify the ``matplotlib`` "backend" in the corresponding Python environment ``matplotlibrc`` file. In the end, I was using ``macosx`` and ``agg`` (on OS X and Ubuntu, respectively). 


### Installing the development branch of ``sklearn`` 

I totally hacked this together (on Ubuntu, didn't test on OS X) and I'm sure there's a more appropriate way to do this. Nevertheless, here's what I did that worked:

**[WIP] do this again to verify** 

- navigate to the appropriate ``site-packages`` directory for your Python environment and clone the ``sklearn`` repo [as described in the docs](http://scikit-learn.org/stable/developers/contributing.html#git-repo).  
 
 


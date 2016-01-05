# MNIST + ``scikit-learn`` = :star2: 


This code was developed for a intra-team Kaggle-like modeling competition on the canonical [MNIST handwritten digits dataset](https://en.wikipedia.org/wiki/MNIST_database).  

We were given the training images and labels, the test images, and a simple Python script that read (and displayed) the binary image data. We had to submit a file with predictions for the test images, one prediction per line.  

The rules were pretty simple: 

- write code in whatever language you want
- use whatever platform you want (laptop, EC2, stone tablet)  
- be prepared to present what you did to the team (in particular, this meant if you cheated and just downloaded the test labels, you were in for A Bad Time) 

All of the data processing and modeling code here is written in Python, and there are a couple of additional bash scripts to facilitate various pieces of the workflow. Most of this code should work out of the box with Python 2.7 on OS X and Ubuntu (an isolated environment is recommended - the requirements.txt I used is included). 

``[placeholder for basic instructions]``

More words about the context and workflow are in [my longer post](www.joshmontague.com). 

-----

## Notes

- I'd suggest using an isolated environment; I used ``virtualenv`` and the Makefile will you set that up 
- As of the time of writing, the development branch of ``sklearn`` is required to use the ``MLPClassifer`` (Multi-layer perceptron). See the additional notes below for instructions on setting this up. 
- This code was developed on both OS X and Ubuntu, which can lead to inconsistencies in the behavior of library behavior. At least once, I had to modify the ``matplotlib`` "backend" in the corresponding Python environment ``matplotlibrc`` file. In the end, I was using ``macosx`` and ``agg`` (on OS X and Ubuntu, respectively). 


### Installing the development branch of ``sklearn`` 

I totally hacked this together and I'm sure there's a more appropriate way to do this. Nevertheless, here's what I did that worked:

**[WIP] do this again to verify** 

- navigate to the appropriate ``site-packages`` directory for your Python environment and clone the ``sklearn`` repo [as described in the docs](http://scikit-learn.org/stable/developers/contributing.html#git-repo).  
- 
  


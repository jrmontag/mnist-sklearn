# Josh Montague, 2015-12
#   MIT License

# locations
#BASEDIR=$(PWD)
#DATADIR=$(BASEDIR)/data
#SAVEDIR=$(BASEDIR)/saved_models
#SUBDIR=$(BASEDIR)/submissions
#LOGDIR=$(BASEDIR)/log


# Python 
BASE_PY=python2
# rename virtualenv if desired
#VENV=tmp-venv
# virtualenv-specific locations
#VBIN=$(BASEDIR)/$(VENV)/bin
# --- maybe we don't need this?
#VPY=$(VBIN)/python

# code 
#CONVERT=convert-binary-data.py
#EXPAND=expand-np-arrays.py


# datetime
DATE := $(shell date +'%Y-%m-%dT%H:%M:%S')
#DATE := $(shell date +'%Y-%m-%d')
TIME := $(shell date +'%H:%M:%S')

help:
	@echo 'Makefile for reproducible analysis                                 '
	@echo '                                                                   '


# run everything in the setup
#all: $(SAVEDIR)/knn_cv-split_*.pdf 
all: saved_models/knn_cv-split_*.pdf 


# example experiment
#$(SAVEDIR)/knn_cv-split_*.pdf: $(DATADIR)/train-images.npy
saved_models/knn_cv-split_*.pdf: data/train-images.npy
	nohup nice bash launch-processes.bash > log/$(DATE)_sample-log.nohup.out & 
	@echo 
	@echo 'Sample experiments are now running in the background.'  
	@echo '... use `tail -f (sample-log) to view overall progress.'  
	@echo '... or use `tail -f (individual logfile) to view individual model progress.'  


# binary data ==> npy arrays
#$(DATADIR)/train-images.npy: $(VBIN)/activate
data/train-images.npy: tmp-venv/bin/activate
	source tmp-venv/activate; \
	python convert-binary-data.py 


# local environment
#$(VBIN)/activate: requirements.txt
tmp-venv/bin/activate: requirements.txt
	#test -d $(VENV) || virtualenv $(VENV) 
	virtualenv -p $(BASE_PY) tmp-venv 
	source tmp-venv/bin/activate ; \
	pip install -r $< 
	touch tmp-venv/bin/activate


# additional training data
#expanded: $(DATADIR)/train-images.npy 
expanded: data/train-images.npy  
	test -e data/expanded-train-images.npy || \
	source tmp-venv/bin/activate ; \
	python expand-np-arrays.py 
	touch $<


# download binary data from web
data/original/train-images.gz: 
	curl http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -o data/original/train-images.gz & 
	curl http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz -o data/original/train-labels.gz & 
	curl http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz -o data/original/test-images.gz  



# !!! delete all generated npy arrays !!! 
clean:
	[ ! -d data ] || rm data/*.npy
   

.PHONY: clean expanded all 

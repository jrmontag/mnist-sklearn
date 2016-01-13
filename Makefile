# Josh Montague, 2015-12
#   MIT License

# locations
BASEDIR=$(PWD)
DATADIR=$(BASEDIR)/data
SAVEDIR=$(BASEDIR)/saved_models
SUBDIR=$(BASEDIR)/submissions
LOGDIR=$(BASEDIR)/log


# Python 
BASE_PY=python2
# rename virtualenv if desired
VENV=tmp-venv
DEVVENV=dev-tmp-venv
# virtualenv-specific locations
VBIN=$(BASEDIR)/$(VENV)/bin

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
demo: $(SAVEDIR)/knn_cv-split_*.pdf 


# example experiment
$(SAVEDIR)/knn_cv-split_*.pdf: $(DATADIR)/train-images.npy
	@echo 
	@echo 'Sample experiments will now run for ~45 seconds. The corresponding '  
	@echo ' log file swill be available in log/ afterward.'  
	@echo 
	@echo 'When complete, cross-validation confusion matrices will open' 
	@echo ' automatically.' 
	nohup nice bash launch-processes.bash > $(LOGDIR)/$(DATE)_sample-log.nohup.out 
	open $(SAVEDIR)/*.pdf 


# binary data ==> npy arrays
$(DATADIR)/train-images.npy: $(VBIN)/activate $(DATADIR)/original/train-images.gz
	. $(VENV)/bin/activate; \
	python convert-binary-data.py 


# local environment
$(VBIN)/activate: requirements.txt
	virtualenv -p $(BASE_PY) $(VENV) 
	. $(VENV)/bin/activate ; \
	pip install -r $< 
	touch $(VENV)/bin/activate


# download binary data from web
$(DATADIR)/original/train-images.gz: 
	curl http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -o data/original/train-images.gz & 
	curl http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz -o data/original/train-labels.gz & 
	curl http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz -o data/original/test-images.gz  


# additional training data
expanded: $(DATADIR)/train-images.npy 
	test -e data/expanded-train-images.npy || \
	. $(VENV)/bin/activate ; \
	python expand-np-arrays.py 
	touch $<


# !!! delete all generated npy arrays !!! 
clean:
	[ ! -d data ] || rm data/*.npy
   

# build env with dev version of sklearn 
skl-dev-env: requirements_dev_sklearn.txt 
	virtualenv -p $(BASE_PY) $(DEVVENV) 
	. $(DEVVENV)/bin/activate ; \
	pip install -r $< 
	touch $(DEVVENV)/bin/activate



.PHONY: clean expanded all 

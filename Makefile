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
# virtualenv-specific locations
VBIN=$(BASEDIR)/$(VENV)/bin
# --- maybe we don't need this?
#VPY=$(VBIN)/python

# code 
CONVERT=convert-binary-data.py
EXPAND=expand-np-arrays.py


# datetime
DATE := $(shell date +'%Y-%m-%dT%H:%M:%S')
#DATE := $(shell date +'%Y-%m-%d')
TIME := $(shell date +'%H:%M:%S')

help:
	@echo 'Makefile for reproducible analysis                                 '
	@echo '                                                                   '


# run everything in the setup
all: $(SAVEDIR)/knn_cv-split_*.pdf 


# example experiment
$(SAVEDIR)/knn_cv-split_*.pdf: $(DATADIR)/train-images.npy
	nohup bash launch-processes.bash > $(LOGDIR)/$(DATE)_sample-log.nohup.out & 
	@echo 
	@echo 'Sample experiments are now running in the background.'  
	@echo '... use `tail -f (sample-log) to view overall progress.'  
	@echo '... or use `tail -f (individual logfile) to view individual model progress.'  


# binary data ==> npy arrays
$(DATADIR)/train-images.npy: $(VBIN)/activate
	source $(VBIN)/activate; \
	python $(CONVERT)


# local environment
$(VBIN)/activate: requirements.txt
	#test -d $(VENV) || virtualenv $(VENV) 
	virtualenv -p $(BASE_PY) $(VENV) 
	source $(VBIN)/activate ; \
	pip install -r $< 
	touch $(VBIN)/activate


# additional training data
expanded: $(DATADIR)/train-images.npy 
	test -e $(DATADIR)/expanded-train-images.npy || \
	source $(VBIN)/activate ; \
	python $(EXPAND)
	touch $<


# !!! delete all created npy arrays !!! 
clean:
	[ ! -d $(DATADIR) ] || rm $(DATADIR)/*.npy
   

.PHONY: clean expanded all 

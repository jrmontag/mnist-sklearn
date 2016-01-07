# Josh Montague, 2015-12
#   MIT License

# locations
BASEDIR=$(PWD)
DATADIR=$(BASEDIR)/data

# Python 
# - specifying 2.7 explicitly leads to an unsolved ImportError 
#BASE_PY=python2.7
BASE_PY=
# rename virtualenv if desired
VENV=tmp-venv
# virtualenv-specific locations
VBIN=$(BASEDIR)/$(VENV)/bin
# --- maybe we don't need this?
VPY=$(VBIN)/python

# code 
CONVERT=convert-binary-data.py
EXPAND=expand-np-arrays.py




# datetime
#DATE := $(shell date +'%Y-%m-%d %H:%M:%S')
DATE := $(shell date +'%Y-%m-%d')
TIME := $(shell date +'%H:%M:%S')

help:
	@echo 'Makefile for reproducible analysis                                 '
	@echo '                                                                   '
	@echo 'Usage:                                                             '
#	@echo '   make everything       build entire project, end-to-end        '
	@echo '   make venv             create local virtualenv for this project (recommended)'
	@echo '   make convert          convert the original binary data into numpy arrays'
	@echo '   make expanded         expand original data (by ~5x) by translating images' 
	@echo '                             (necessary for best leaderboard score)'
	@echo '                                                                   '
	@echo '   make clean            remove all the generated data files       '
	@echo '                                                                   '


# update this later, use one of the first handful of submissions as the 
#   dependency
#all: submissions/*-foo.submission

#submissions/*-foo.submission: $(DATADIR)/test-images.npy
#   run bash process script (leave commit w/ a few experiments in seq) 


$(DATADIR)/test-images.npy: $(VENV)/bin/activate
	source $(VBIN)/activate; \
	python $(CONVERT)


$(VENV)/bin/activate: requirements.txt
	#test -d $(VENV) || virtualenv -p $(BASE_PY) $(VENV) 
	test -d $(VENV) || virtualenv $(VENV) 
	$(VBIN)/pip install -r $< 
	touch $(VBIN)/activate






# target for building dev sklearn?


clean:
	[ ! -d $(DATADIR) ] || rm $(DATADIR)/*.npy
   


## old
#
#expanded: convert 
#	test -e $(DATADIR)/expanded-train-images.npy || $(VPY) $(EXPAND) 
#
#


.PHONY: venv clean convert everything 

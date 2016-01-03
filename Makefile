# Josh Montague, 2015-12

# locations
BASEDIR=$(PWD)
DATADIR=$(BASEDIR)/data
#BINDIR=$(BASEDIR)/bin

##### Python details
# choose your favorite system interpreter
BASE_PY=python2.7
# rename virtualenv if desired
VENV=tmp-venv
# virtualenv-specific locations
VBIN=$(BASEDIR)/$(VENV)/bin
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


#everything: convert 


expanded: convert 
	test -e $(DATADIR)/expanded-train-images.npy || $(VPY) $(EXPAND) 


convert: venv 
	test -e $(DATADIR)/\*-images.npy || $(VPY) $(CONVERT)


venv: $(VENV)/bin/activate


$(VENV)/bin/activate: requirements.txt
	test -d $(VENV) || virtualenv -p $(BASE_PY) $(VENV) 
	$(VBIN)/pip install -Ur requirements.txt
	touch $(VBIN)/activate
#	# this part needs to be figured out still
#	cd $(VENV)/lib/python2.7/site-packages/scikit-learn; \
#    source $(VBIN)/activate; \
#    python setup.py build_ext --inplace; \
#    cd ..; ln -s scikit-learn/sklearn sklearn; \
#    cd $(BASEDIR) 


clean:
	[ ! -d $(DATADIR) ] || rm $(DATADIR)/*.npy
   






.PHONY: venv clean convert everything 

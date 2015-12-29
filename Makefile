# Josh Montague, 2015-12

# locations
BASEDIR=$(PWD)
DATADIR=$(BASEDIR)/data
#BINDIR=$(BASEDIR)/bin

# files
CONVERT=convert-binary-data.py

##### Python details
# choose your favorite system interpreter
BASE_PY=python2.7
# rename virtualenv if desired
VENV=tmp-venv
# virtualenv-specific locations
VBIN=$(BASEDIR)/$(VENV)/bin
VPY=$(VBIN)/python



# datetime
#DATE := $(shell date +'%Y-%m-%d %H:%M:%S')
DATE := $(shell date +'%Y-%m-%d')
TIME := $(shell date +'%H:%M:%S')

help:
	@echo 'Makefile for reproducible analysis                                 '
	@echo '                                                                   '
	@echo 'Usage:                                                             '
	@echo '   make everything       build entire project, end-to-end        '
	@echo '   make convert          convert the original binary data into numpy arrays  '
	@echo '   make venv             create local virtualenv for this project '
	@echo '                                                                   '
	@echo '   make clean            remove all the generated data files       '
	@echo '                                                                   '


everything: convert 


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

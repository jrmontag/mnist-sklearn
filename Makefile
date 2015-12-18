# Josh Montague, 2015-12

# locations
BASEDIR=$(CURDIR)
DATADIR=$(BASEDIR)/data
BINDIR=$(BASEDIR)/bin

# files
CONVERT=convert-binary-data.py

##### Python details
# choose your favorite system interpreter
BASE_PY=python
# rename virtualenv if desired
VENV=tmp-venv
VPY=$(BASEDIR)/$(VENV)/bin
PY=$(VENV)/bin/python



# datetime
#DATE := $(shell date +'%Y-%m-%d %H:%M:%S')
DATE := $(shell date +'%Y-%m-%d')
TIME := $(shell date +'%H:%M:%S')

help:
	@echo 'Makefile for reproducible analysis                                 '
	@echo '                                                                   '
	@echo 'Usage:                                                             '
	@echo '   make venv             create local virtualenv for this project '
	@echo '   make clean            remove all the generated data files       '
	@echo '   make convert          convert the original binary data into numpy arrays  '
	@echo '                                                                   '
	@echo '                                                                   '


clean:
	[ ! -d $(DATADIR) ] || rm $(DATADIR)/*.npy
   


venv: $(VENV)/bin/activate

$(VENV)/bin/activate: requirements.txt
	test -d $(VENV) || virtualenv -p $(BASE_PY) $(VENV) 
	$(VENV)/bin/pip install -Ur requirements.txt
	touch $(VENV)/bin/activate

convert: venv 
	[ ! -e $(DATADIR) ] || $(PY) $(BINDIR)/$(CONVERT)


everything: convert 







.PHONY: venv clean convert everything 

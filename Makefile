# Josh Montague, 2015-12

# commands
VENV_ACT=/Users/jmontague/.virtualenvs/data-2.7-tw/bin/activate
PY=python

# locations
BASEDIR=$(CURDIR)
DATADIR=$(BASEDIR)/data

# files
CONVERT=convert-binary-data.py


# datetime
#DATE := $(shell date +'%Y-%m-%d %H:%M:%S')
DATE := $(shell date +'%Y-%m-%d')
TIME := $(shell date +'%H:%M:%S')

help:
	@echo 'Makefile for reproducible analysis                                 '
	@echo '                                                                   '
	@echo 'Usage:                                                             '
	@echo '   make clean            remove all the generated data files       '
	@echo '   make convert          convert the original binary data into numpy arrays  '
	@echo '                                                                   '
	@echo '                                                                   '


clean:
	[ ! -d $(DATADIR) ] || rm $(DATADIR)/*.npy
   
convert:
	source $(VENV_ACT); \
	[ ! -e $(DATADIR) ] || $(PY) $(BASEDIR)/$(CONVERT); \
	deactivate






tmp-venv: tmp-venv/bin/activate
tmp-venv/bin/activate: requirements.txt
	virtualenv tmp-venv
	tmp-venv/bin/pip install -Ur requirements.txt
	touch tmp-venv/bin/activate






.PHONY: venv clean convert 

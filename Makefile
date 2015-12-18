# Josh Montague, 2015-12

# commands
PY=python

# locations
BASEDIR=$(CURDIR)
DATADIR=$(BASEDIR)/data

# files
HDF5_FILE=$(DATADIR)/data.hdf5

# datetime
#DATE := $(shell date +'%Y-%m-%d %H:%M:%S')
DATE := $(shell date +'%Y-%m-%d')
TIME := $(shell date +'%H:%M:%S')

help:
	@echo 'Makefile for reproducible analysis                                 '
	@echo '                                                                   '
	@echo 'Usage:                                                             '
	@echo '   make venv             create a new virtualenv for this work     '
	@echo '   make clean            remove all the generated data files       '
	@echo '   make convert          convert the original binary data into numpy arrays  '
	@echo '                                                                   '
	@echo '                                                                   '


venv:
	@echo '...WIP...'
 
clean:
	[ ! -d $(DATADIR) ] || rm $(DATADIR)/*.npy
   
convert:
	@echo '...WIP...'
	#[ ! -e $(HDF5_FILE) ] || $(PY) $(BASEDIR)/image.py 


.PHONY: venv clean convert 

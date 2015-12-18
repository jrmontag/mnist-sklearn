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
	@echo '   make clean            remove the generated data files           '
	@echo '   make venv             create a new virtualenv for this work     '
	@echo '   make hdf5             create a new virtualenv for this work     '
	@echo '                                                                   '
	@echo '                                                                   '


venv:
	@echo '...WIP...'
 
clean:
	[ ! -d $(DATADIR) ] || rm $(DATADIR)/*.npy
   
hdf5:
	[ ! -e $(HDF5_FILE) ] || $(PY) $(BASEDIR)/image.py 


.PHONY: html help clean regenerate serve devserver publish ssh_upload rsync_upload dropbox_upload ftp_upload s3_upload cf_upload github newpost newpage savedraft

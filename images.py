#!/usr/bin/env python

# 
# original by Scott Hendrickson, modified by Josh Montague 
#

import fileinput
import logging
import numpy as np
import struct


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', 
                    #level=logging.DEBUG
                    level=logging.INFO
                    )

# switch outputs
PLOT = False 

######### read the labels from binary file
logging.info("reading training labels")
label_names = ["data/train-labels.gz"]
g = fileinput.FileInput(label_names, openhook=fileinput.hook_compressed)
# grab the first chunk of data for header info
logging.info(" reading header")
x = g.next()
head = []
for i in range(2):
    head.append(struct.unpack(">I", x[4*i:4*i+4])[0])
magic, n_labels = head
logging.info(" magic={}, labels={}".format(*head))

# unsigned binary ints - 1 byte each
logging.info(" reading data")
labels = []
j = 8 # byte index on current chunk
while len(labels) < n_labels:
    try:
        val = struct.unpack("B", x[j])[0]
    except IndexError:
        # read a new chuck from file
        x = g.next()
        j = 0
        val = struct.unpack("B", x[j])[0]
    labels.append(val)
    j += 1
#logging.debug("labels: {}".format(labels))
label_array = np.array(labels)
logging.debug(" label_array type: {} (length: {})".format(type(label_array), len(label_array)))

logging.info("writing numpy label array to disk") 
with open('data/train-labels.npy', 'wb') as f:
    np.save(f, label_array)

########## read images from binary file
logging.info("reading training data")
infile_names = ["data/train-images.gz"]
f = fileinput.FileInput(infile_names, openhook=fileinput.hook_compressed)
x = f.next()
head = []
logging.info(" reading header")
for i in range(4):
    head.append(struct.unpack(">I", x[4*i:4*i+4])[0])
magic, n_images, rows, columns = head
logging.info(" magic={}, images={}, rows={}, cols={}".format(*head))
# now we know the shape of the data, so we can allocate an array
data_array = np.zeros((n_images, rows*columns), dtype=int)

logging.info(" reading data")
j = 16 # index in current chunk
for i in range(n_images):
    # i loop is over the samples 
    sample_i_values = []
    for r in range(rows):
        # keep appending to sample array all the way through 
        #   the rows and cols of sample i
        for c in range(columns):
            try:
                val = struct.unpack("B", x[j])[0]
            except IndexError:
                # need to read a new chunck of data from finle
                x = f.next()
                j = 0
                val = struct.unpack("B", x[j])[0]
            if PLOT:
                ##################################
                # simple image plots using screen text layout
                # 3 levels of grey
                if val > 170:
                    print "#",
                elif val > 85:
                    print ".",
                else:
                    print " ",
                ##################################
            # append this value to the sample row 
            sample_i_values.append(val)
            j += 1
        if PLOT:
            print "row={:2}, j={:4}".format(r,j)
    if PLOT:
        print "image={}, label={}".format(i, labels[i])
    # visually verify that our numeric values are similar to the ascii art
    logging.debug("sample_i_values (len={}): {}".format(len(sample_i_values), sample_i_values))

    # update the row corresponding to this image 
    data_array[i] = np.array(sample_i_values)

logging.info("writing numpy image array to disk") 
with open('data/train-images.npy', 'wb') as f:
    np.save(f, data_array)


    
                
    

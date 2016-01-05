#!/usr/bin/env python
import fileinput
import struct

# read the labels from binary file
label_names = ["train-labels.gz"]
g = fileinput.FileInput(label_names, openhook=fileinput.hook_compressed)
# grab the first chunk of data for header info
x = g.next()
head = []
for i in range(2):
    head.append(struct.unpack(">I", x[4*i:4*i+4])[0])
magic, n_labels = head
print "magic={}\nlabels={}".format(*head)
# start reading the lables
# unsigned binary ints - 1 byte each
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

# read images from binary file
infile_names = ["train-images.gz"]
f = fileinput.FileInput(infile_names, openhook=fileinput.hook_compressed)
x = f.next()
head = []
for i in range(4):
    head.append(struct.unpack(">I", x[4*i:4*i+4])[0])
magic, n_images, rows, columns = head
print "magic={}\nimages={}\nrows={}\ncols={}".format(*head)
j = 16 # index in current chunk
for i in range(n_images):
    for r in range(rows):
        for c in range(columns):
            try:
                val = struct.unpack("B", x[j])[0]
            except IndexError:
                # need to read a new chunck of data from finle
                x = f.next()
                j = 0
                val = struct.unpack("B", x[j])[0]
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
            j += 1
        print "row={:2}, j={:4}".format(r,j)
    print "image={}, label={}".format(i, labels[i])
                
    

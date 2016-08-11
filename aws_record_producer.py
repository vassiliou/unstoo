from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os.path
import glob
from skimage import io
from skimage import measure


### Some flags:

on_mac = False

kaggle = False

###

INPUT_DIRECTORY = '/home/ubuntu/train'

RECORD_DIRECTORY = '/home/ubuntu/train'

if on_mac:
    INPUT_DIRECTORY = '/Users/gus/Desktop/aws/3_validation_inputs'
    RECORD_DIRECTORY = '/Users/gus/Desktop/aws/3_validation_records'  

IMG_PATTERN = '*.tif'

DOWNSAMPLE_FACTOR = 2

pattern =  os.path.join(INPUT_DIRECTORY, IMG_PATTERN)

print(pattern)

filepaths = sorted(glob.glob(pattern))

img_paths = sorted([f for f in filepaths if 'mask' not in f])


records = img_paths

for rec in records:
    mask_path = rec.replace('.tif', '_mask.tif')
    img = io.imread(rec)
    img = measure.block_reduce(img,(DOWNSAMPLE_FACTOR,DOWNSAMPLE_FACTOR),func=np.mean).astype(np.uint8)
    if kaggle == True:
        mask = np.zeros_like(img)
    else:
        mask = (io.imread(mask_path)/255).astype(np.uint8)
        mask = measure.block_reduce(mask,(DOWNSAMPLE_FACTOR,DOWNSAMPLE_FACTOR),func=np.max)
    img = img.flatten()
    mask = mask.flatten()
    flat_record = np.concatenate((img,mask))
    f_name = os.path.split(rec)[1]
    f_name = os.path.splitext(f_name)[0]+'.rec'
    write_path = os.path.join(RECORD_DIRECTORY,f_name)
    flat_record.astype('uint8').tofile(write_path)
    #print('{file}'.format(file=write_path))


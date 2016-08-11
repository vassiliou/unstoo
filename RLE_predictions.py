import numpy as np
from skimage import transform

import os
import glob


###

DECISION_BOUNDARY = 0.5

AREA_CUTOFF = 4000


output_pattern='/home/ubuntu/test_output/*.npy'
fnames = glob.glob(output_pattern)
indices = np.argsort([int(f.split('/')[-1].split('_')[0]) for f in fnames])
orderedfnames = [fnames[i] for i in indices]


def to_RLE(mask):
        """Convert mask to run length encoded format"""
        dm = np.diff(mask.T.flatten().astype('int16'))
        start = np.nonzero(dm>0)[0]
        stop = np.nonzero(dm<0)[0]
        RLE = np.vstack((start+2, stop-start))
        return ' '.join(str(n) for n in RLE.flatten(order='F'))


RLE = []


for i, file in enumerate(orderedfnames, 1):
    # Load the prediction file
    raw = np.load(file)[0,:,:]
    upsized = transform.pyramid_expand(raw,2)
    # Apply masking function
    prediction = upsized > DECISION_BOUNDARY
    if i==1:
        print prediction.shape
    area = prediction.sum()
    # Apply size-based cutoff and output RLE
    if area < AREA_CUTOFF:
        RLE.append((i,''))
    else:
        RLE.append((i,to_RLE(prediction)))


out = '\n'.join('{i},{rle}'.format(i=r[0],rle=r[1]) for r in RLE)
with open('output.csv','w') as f:
    f.write('img,pixels\n')
    f.write(out)

from skimage import io
from skimage import measure
from skimage import morphology

from scipy.interpolate import InterpolatedUnivariateSpline

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import os
#import glob
import pandas as pd
import numpy as np


CM = plt.cm.inferno(np.arange(256))
alpha =  np.linspace(0, 1, 256)
CM[:,-1] = alpha
CM = ListedColormap(CM)

datafolder = "/Users/gus/CDIPS/nerve-project/"
trainbin = "/Users/gus/CDIPS/uns/training.bin"

if os.environ['USER'] == 'chrisv':
    print(os.environ['USER'], end='')
    try:
        session =  os.environ['SESSION']
    except KeyError:
        session = 'mac'
    finally:
        if session == 'Lubuntu':
            print(" on Lubuntu")
            datafolder = '/home/chrisv/code'
            trainbin = '/home/chrisv/code/uns/training.bin'
            bottlefolder = '/home/chrisv/code/bottleneck_files'
        else:
            print(" on Mac")
            trainbin = '/Users/chrisv/Code/CDIPS/uns/training.bin'
            datafolder = "/Users/chrisv/Code/CDIPS"
#            for k,v in sorted(os.environ.items()):
#                print((k,v))

# Usage:
# image_pair(sub_im(subject,image))
sub_im = lambda subject, img: pd.Series(data=[subject, img], index=['subject', 'img'])


trainfolder = os.path.join(datafolder, 'train')
testfolder = os.path.join(datafolder, 'test')

training = pd.read_msgpack(trainbin)

def dice(preds, truth):
    numer = 2*np.sum(np.logical_and(preds, truth))
    denom = np.sum(preds) + np.sum(truth)
    if denom < 1:
        score = 1  # denom==0 implies numer==0
    else:
        score = numer/denom
    return score


class image():
    def __init__(self, row):
        if type(row) is np.ndarray:
            # given an array, assume this is the image
            self._image = row
            self.title = ''
            self.filename = ''
        else:
            self.info = row
            self._image = None  # io.imread(os.path.join(trainfolder, imagefile))
            self.title = '{subject}_{img}'.format(subject=row['subject'],
                                                  img=row['img'])
            self.filename = self.title + '.tif'

    def __str__(self):
        return self.info.__str__()

    def __repr__(self):
        return self.info.__repr__()


    def load(self):
        """ Load image file """
        return io.imread(os.path.join(trainfolder, self.filename))

    def load_rgb(self, trim=2):
        if trim>0:
            grayscale = self.image[trim:-trim,trim:-trim]
        else:
            grayscale = self.image
        return np.dstack((grayscale,grayscale,grayscale))

    def plot(self, ax=None, **plotargs):
        if ax is None:
            fig, ax = plt.subplots()
        plotargs['cmap'] = plotargs.get('cmap',plt.cm.gray)
        ax.imshow(self.image, **plotargs)
        ax.set_title(self.title)
        ax.axis('equal')
        ax.axis('off')
        ax.tick_params(which='both', axis='both',
                          bottom=False, top=False, left=False, right=False,
                          labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        ax.autoscale(tight=True)

        return ax

    @property
    def image(self):
        if self._image is None:
            self._image = self.load()
        return self._image

    def get_patch(image,pixel,F):
            hor_range = (pixel[0]-F,pixel[0]+F+1)
            ver_range= (pixel[1]-F,pixel[1]+F+1)
            return image[hor_range[0]:hor_range[1],ver_range[0]:ver_range[1]]

class prediction(image):
    def __init__(self, filename, untrim=2):
        self.title = ''
        self.filename = filename
        self._image = None
        self.info = 'Prediction'
        self.untrim = untrim
        self.predmasks = {}

    def load(self):
        """ Load prediction file and expand by untrim"""
        pred_image = np.load(self.filename)
        if self.untrim > 0:
            u = self.untrim
            w, h = pred_image.shape
            pred_array = np.zeros((w+2*u, h+2*u))
            pred_array[u:-u,u:-u] = pred_image
            return pred_array
        else:
            return pred_image

    def new_prediction(self, key, maskfun=lambda x:x>0.5):
        """ Give it a name for referencing and a function to apply to the prediction probablility"""
        predmask = maskfun(self.image)
        self.predmasks[key] = mask(predmask*255)
        #self.scores[key] = dice(predmask, self.boolmask)

    def blank_prediction(self):
        self.scores['blank'] = dice(np.zeros(self.boolmask.shape), self.boolmask)

    def heatmap(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
            colormap = plt.cm.inferno
        else:
            colormap = CM
        self.plot(ax=ax, vmin=0, vmax=1.0, cmap=colormap)
        for k, pred in self.predmasks.items():
           pred.plot_contour(ax=ax, label=k)

class mask(image):
    def __init__(self, info):
        image.__init__(self, info)

        self._contour = None
        self.contourlength = 40
        self.filename = self.title + '_mask.tif'
        self._properties = None
        self.hasmask = False

    @property
    def contour(self):
        if self._contour is None:
            contours = measure.find_contours(self.image, 254.5)
            # downsample contour
            if len(contours)>0:
                contour = contours[np.argmax([c.shape[0] for c in contours])]
                T_orig = np.linspace(0, 1, contour.shape[0])
                ius0 = InterpolatedUnivariateSpline(T_orig, contour[:,0])
                ius1 = InterpolatedUnivariateSpline(T_orig, contour[:,1])
                T_new = np.linspace(0, 1, self.contourlength)
                self._contour = np.vstack((ius0(T_new), ius1(T_new)))
                self.hasmask = True
            else:
                self.hasmask = False
        return self._contour

    @contour.setter
    def contour(self, contour):
        self._contour = contour

    def one_hot(self, trim=2):
        if trim>0:
            raw_mask=self.image[trim:-trim,trim:-trim].astype(bool)
        else:
            raw_mask = self.image.astype(bool)
        return np.dstack((raw_mask, ~raw_mask)).astype(np.float32)

    @property
    def properties(self):
        """Return a set of metrics on the masks with units of distance """
        imgH, imgW = self.image.shape  # Image height, width

        # Don't overemphasize one dimension over the other by setting the max
        # dimenstion to equal 1
        imgL = np.max([imgH, imgW])
        imgA = imgH * imgW  # Total number of pixels
        if self._properties is None:
            # Must load contour into single variable before checking self.hasmask
            # If mask exists, only then can we access x,y components of contour
            C = self.contour
            if self.hasmask:
                D = {}
                D['hasmask'] = True


                # Area metric is normalize to number of image pixels.  Sqrt
                # converts units to distance
                D['maskarea'] = np.sqrt(np.count_nonzero(self.image)/imgA)
                # Contour-derived values
                x, y = C
                D['contxmin'] = np.min(x)/imgL
                D['contxmax'] = np.max(x)/imgL
                D['contymin'] = np.min(y)/imgL
                D['contymax'] = np.max(y)/imgL

                D['contW'] = D['contxmax'] - D['contxmin']
                D['contH'] = D['contymax'] - D['contymin']

                # Image moments
                m = measure.moments(self.image, order=5)
                D['moments'] = m
                D['centrow'] = (m[0, 1]/m[0, 0])/imgL
                D['centcol'] = (m[1, 0]/m[0, 0])/imgL

                # Hu, scale, location, rotation invariant (7, 1)
                mHu = measure.moments_hu(m)
                for i, Ii in enumerate(mHu):
                    D['moment_hu_I{}'.format(i)] = Ii

                # Contour SVD is converted to two coordinates
                # First normalize and centre the contours
                D['contour'] = self.contour

                contour = (self.contour.T/imgL - [D['centrow'], D['centcol']]).T
                D['unitcontour'] = contour

                _, s, v = np.linalg.svd(contour.T)

                D['svd'] = s*v
                D['svdx0'] = D['svd'][0,0]
                D['svdx1'] = D['svd'][0,1]
                D['svdy0'] = D['svd'][1,0]
                D['svdy1'] = D['svd'][1,1]

                # Width by medial axis
                skel, distance = morphology.medial_axis(self.image,
                                                        self.image,
                                                        return_distance=True)
                self.skel = skel
                self.distance = distance

                #
                D['skelpixels'] = np.sqrt((np.sum(skel)/imgA))  # number of pixels

                # distances should be restricted to within mask to avoid over-
                # counting the zeros outside the mask
                distances = distance[self.image>0]/imgL
                q = [10, 25, 50, 75, 90]
                keys = ['skeldist{:2d}'.format(n) for n in q]
                vals = np.percentile(distances, q)
                D.update(dict(zip(keys, vals)))
                D['skelavgdist'] = np.mean(distances)
                D['skelmaxdist'] = np.max(distances)

                self._properties = D
        return self._properties

    @property
    def pandas(self):
        df = self.info[['subject','img','pixels']]
        df.loc['hasmask'] = False
        props = self.properties
        if props is not None:
            for k, v in props.items():
                df.loc[k] = v
        return df

    @property
    def RLE(self):
        """Convert mask to run length encoded format"""
        dm = np.diff(self.image.T.flatten().astype('int16'))
        start = np.nonzero(dm>0)[0]
        stop = np.nonzero(dm<0)[0]
        RLE = np.vstack((start+2, stop-start))
        return ' '.join(str(n) for n in RLE.flatten(order='F'))


    def plot_contour(self, *args, **kwargs):
        C = self.contour
        if self.hasmask:
            x = C[1,:]
            y = C[0,:]
            ax = kwargs.pop('ax', None)
            if ax is None:
                fig, ax = plt.subplots()
            ax.plot(x,y, *args, **kwargs)
            ax.axis('equal')
            ax.tick_params(which='both', axis='both',
                              bottom=False, top=False, left=False, right=False,
                              labelbottom=False, labeltop=False, labelleft=False, labelright=False)
            ax.autoscale(tight=True)
            return ax
        else:
            return None


class image_pair(object):
    def __init__(self, row=None, pred=None):
        self.image = image(row)
        self.mask = mask(row)
        self.boolmask = self.mask.image.astype(bool)
        self.predmasks = {}
        self.scores = {}

        self.bottlefile = self.image.title + '.btl'
        if pred is None:
            self.predfile = '{}_pred.tif'.format(self.image.title)
        else:
            self.predfile = pred.format(self.image.title)
            self.pred=prediction(self.predfile)


    def plot(self, ax=None):
        if ax is None:
            ax = self.image.plot()
        else:
            self.image.plot(ax=ax)
        self.mask.plot_contour(ax=ax, c='r', lw=2.0, label="Ground truth")
        return ax


    def plot_predictions(self):
        fig, ax = plt.subplots(1,2,figsize=(16,6))
        self.plot(ax=ax[0])
        self.pred.plot(ax=ax[1], vmin=0, vmax=1.0, cmap=plt.cm.inferno)
        for k, pred in self.predmasks.items():
            pred.plot_contour(ax=ax[0], lw=2.0)
#            pred.plot_contour(ax=ax[1], label=k)

    def plot_heatmap(self):
        ax = self.image.plot()
        self.pred.plot(ax=ax, vmin=0, vmax=1.0, cmap=CM)
        self.mask.plot_contour(ax=ax, c='r', label="Ground truth")
        for k, pred in self.predmasks.items():
           pred.plot_contour(ax=ax, label=k)


def plot_pca_comps(P, ncomp, *args, **kwargs):
    fig = plt.figure()
    for i in np.arange(ncomp):
        for j in np.arange(i, ncomp):
            ax = fig.subplot(ncomp-1, ncomp-1, j+i*(ncomp-1))
            ax.scatter(P[:,i], P[:,j], *args, **kwargs)

class batch(list):
    def __init__(self, rows, pred=None):
        list.__init__(self, [])
        for row in rows.iterrows():
            self.append(image_pair(row[1], pred))

    @property
    def array(self):
        """ Load a series of images and return as a 3-D numpy array.
        imageset consists of rows from training.bin"""
        return np.array([im.image.image for im in self])

    def array_masks(self, trim=2):
        """Load masks from the batch into a 4-D ndarray"""
        entries=[]
        for impair in self:
            entries.append(impair.mask.one_hot(trim=trim))
        return np.array(entries).astype(np.float32)

    def array_rgb(self, trim=2):
        """ Load a series of images and return as a 3-D numpy array.
        imageset consists of rows from training.bin"""
        return np.array([im.image.load_rgb(trim=trim) for im in self])

    def plot_grid(self, ncols=5, plotimage=True, plotcontour=True, plotpred=False, figwidth=16):
        """Plot a grid of images, optionally overlaying contours and predicted contours
            Assumes the input is a Pandas DataFrame as in training.bin
        """
        nrows=int(np.ceil(len(self)/ncols))
        figheight = figwidth/ncols*nrows
        fig = plt.figure(figsize=(figwidth,figheight))
        for idx, imgpair in enumerate(self,1):
            ax = fig.add_subplot(nrows, ncols, idx)
            if plotimage:
                imgpair.image.plot(ax=ax)
            if plotcontour:
                imgpair.mask.plot_contour('-b', ax=ax)
            if plotpred:
                imgpair.pred.plot_contour('-r', ax=ax)

        return ax


    def plot_hist(self, ax=None):
        """Plot histograms of a set of images
            Assumes the input is a Pandas DataFrame as in training.bin
        """
        if ax is None:
            fig, ax = plt.subplots()
        for imgpair in self:
            ax.hist(imgpair.image.image.flatten(), cumulative=True, normed=True,
                    bins=100, histtype='step', label=imgpair.image.filename, alpha=0.1, color='k')
        return ax

    def scores(self):
        scores = [imgpair.score for imgpair in self]
        return scores

if __name__ == '__main__':

    # Load image pair from training table
    img = image_pair(training.iloc[0])

    #plot individual image/maks
    fig, ax = plt.subplots(1,2)
    img.image.plot(ax=ax[0])
    img.mask.plot(ax=ax[1])

    #plot image pair to overlay contour
    img.plot()

    #create/plot batch of images
    imgbatch = batch(training.iloc[0:6])
    imgbatch.plot_grid()

    #use batch.array to get a NumPy array of all images in a batch
    #Call image with a 2-D NumPy array to get access to plotting etc.
    imgsum = image(np.sum(imgbatch.array,axis=0))
    imgsum.plot(cmap=plt.cm.viridis)

    # histograms of batch of images?
    imgbatch.plot_hist()
    plt.show()

    print(img.mask.properties)
    print(img.mask.pandas)

    # Check that properties are called/set correctly
    img = image_pair(training.iloc[1])
    print(img.mask.pandas)

    # Use batch.pop() to process images sequentially

#    import psutil
#    process = psutil.Process(os.getpid())
#    # Memory usage
#    newbatch=[]
#    mem = process.memory_info().rss
#    newbatch = batch(training.iloc[10:16])
#    print('Batch of 6: {:d}'.format(process.memory_info().rss))
#    newbatch.array
#    print('Batch of 6 with images: {:d}'.format(process.memory_info().rss))
#    mem = process.memory_info().rss
#    newbatch = batch(training.iloc[0:50])
#    A = newbatch.pop().image.image
#    for i in range(len(newbatch)):
#        A = A + newbatch.pop().image.image
#        if i%10 == 0:
#            print('{:d}, images: {:d}'.format(i,process.memory_info().rss))
#    savebatch = batch(training.iloc[0:50])
#    for i, img in enumerate(savebatch):
#        data = img.image.image
#        if i%10 == 0:
#            print('{:d}, images: {:d}'.format(i,process.memory_info().rss))
#
#    print(len(newbatch), len(savebatch))

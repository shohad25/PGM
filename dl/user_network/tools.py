import cPickle
from matplotlib import pyplot

from numpy import flipud,shape,zeros,rot90,ceil,sqrt
import pylab
from src import network
from src.network import Network
from user_network import *

def montage(X, colormap=pylab.cm.gist_gray):
    m, n, count = shape(X)
    mm = int(ceil(sqrt(count)))
    nn = mm
    M = zeros((mm * m, nn * n))

    image_id = 0
    for j in range(mm):
        for k in range(nn):
            if image_id >= count:
                break
            sliceM, sliceN = j * m, k * n
            M[sliceN:sliceN + n, sliceM:sliceM + m] = X[:, :, image_id]
            image_id += 1

    pylab.imshow(flipud(rot90(M)), cmap=colormap)
    pylab.axis('off')
    return M

def dump_pkl(obj, fp):
    f = open(fp, 'w+')
    cPickle.dump(obj, f)
    f.close()

def load_pkl(fp):
    f = open(fp, 'r+')
    retval = cPickle.load(f)
    f.close()
    return retval

def imshow(im, title='No Title'):
    pyplot.figure()
    pyplot.imshow(im)
    pyplot.draw()
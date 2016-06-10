from os.path import dirname, join
import sys

try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np


def _safe_unpickle(file_name):
    with open(file_name, "rb") as data_file:
        if sys.version_info >= (3, 0):
            # python3 unpickling of python2 unicode
            data = pickle.load(data_file, encoding="latin1")
        else:
            data = pickle.load(data_file)
    return data


def load_letters():
    """Load the OCR letters dataset.

    This is a chain classification task.
    Each example consists of a word, segmented into letters.
    The first letter of each word is ommited from the data,
    as it was a capital letter (in contrast to all other letters).


    References
    ----------
    http://papers.nips.cc/paper/2397-max-margin-markov-networks.pdf
    http://groups.csail.mit.edu/sls/archives/root/publications/1995/Kassel%20Thesis.pdf
    http://www.seas.upenn.edu/~taskar/ocr/
    """
    module_path = dirname(__file__)
    data = _safe_unpickle(join(module_path, 'letters.pickle'))
    # we add an easy to use image representation:
    data['images'] = [np.hstack([l.reshape(16, 8) for l in word])
                      for word in data['data']]

    return data


def mnist_loader():
    from common.datasets.mnist_loader import load_data_wrapper
    return load_data_wrapper()
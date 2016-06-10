import os, sys
import random
import cPickle

from src.network import Network
from src import network as network
from src import mnist_loader
from user_network import analyzer
import numpy as np

# create analyzer
my_analyzer = analyzer.Analyzer()
# set training's data
basic_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

# list_of_dicts = ['sobel', 'laplacian', 'canny', 'harris', 'basic2']
list_of_dicts = ['basic2']
prediction_size = 10

mode = 'Train'

training_lengths = [50000, 40000, 30000, 20000, 10000, 5000, 1000, 500, 100, 10]

predictions = {length: [] for length in training_lengths}

for dic_name in list_of_dicts:
    training_directory = os.path.join(basic_dir, "outputs/NN_" + dic_name)
    if mode == 'Train':
        f = open(os.path.join(training_directory, 'predictions_norm.pkl'), 'r+')
    else:
        f = open(os.path.join(training_directory, 'predictions_test_norm.pkl'), 'r+')
    m_pred = cPickle.load(f)
    f.close()
    #  add to total predictions
    for (key, val) in m_pred.iteritems():
        predictions[key].append(val)
    print "Update {0}".format(str(dic_name))


all_data = {}

for (length, ftrs) in predictions.iteritems():
    X = []
    f = ftrs[0]
    Y = np.array(zip(*ftrs[0])[1])

    for ftr in ftrs:
        ftr_pred = np.array(zip(*ftr)[0]).transpose()
        X.append(ftr_pred)

    X = np.array(X)
    if mode == 'Train':
        X = X.reshape((len(list_of_dicts) * prediction_size, length))
    else:
        X = X.reshape((len(list_of_dicts) * prediction_size, 10000))
    X = X.transpose()
    X_final = []
    for x in X:
        X_final.append(x[:, np.newaxis])

    X_final = np.array(X_final)
    all_data.setdefault(length, zip(X_final, Y))

if mode == 'Train':
    save_directory = os.path.join(basic_dir, "data/basic_nn_predictions_norm_train.pkl")
else:
    save_directory = os.path.join(basic_dir, "data/basic_nn_predictions_norm_test.pkl")
f = open(save_directory, 'w+')
cPickle.dump(all_data, f)
f.close()
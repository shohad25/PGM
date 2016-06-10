import os, sys
import random
import cPickle
import numpy as np
from dl.src.network import Network
from common.datasets.load_datasets import mnist_loader, load_letters
from dl.user_network import *
import pdb

# get the data
letters = load_letters()
X, y, folds = letters['data'], letters['labels'], letters['folds']

# we convert the lists to object arrays, as that makes slicing much more
# convenient
X, y = np.array(X), np.array(y)
X_train, X_test = X[folds == 1], X[folds != 1]
y_train, y_test = y[folds == 1], y[folds != 1]

# separate words to letters:
X_train = np.vstack(X_train)
X_test = np.vstack(X_test)
y_train = np.hstack(y_train)
y_test = np.hstack(y_test)

# Convert to mnist format:
training_data = []
for i, n in enumerate(y_train):
    n_vec = np.zeros((26,1))
    n_vec[n] = 1
    training_data.append((X_train[i], n_vec))

test_data = []
for i, n in enumerate(y_test):
    n_vec = np.zeros((26,1))
    n_vec[n] = 1
    test_data.append((X_test[i], n_vec))

dict_base_name = "training_"

# initialize a network for training
image_size = 128
num_of_labels = 26
num_of_hidden_layers = 30
net = Network([image_size, num_of_hidden_layers, num_of_labels])
# train NN:
num_of_epochs = 30
mini_batch_size = 10
eta = 3.0
net.SGD(training_data, num_of_epochs, mini_batch_size, eta, test_data)

# save results:
#base_path = "/home/ohadsh/Dropbox/Rami/Code_ohad/outputs/NN_basic"
#train_path = os.path.join(base_path, dict_base_name)
#os.makedirs(train_path)
#training_results_handler.save_net(net, os.path.join(train_path, "net.pcl"))
#training_results_handler.save_net_params(net.sizes, len(training_data),
#                                         len(test_data), num_of_epochs, mini_batch_size,
#                                     eta, net.test_results, os.path.join(train_path, "info.txt"))

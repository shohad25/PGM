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
    training_data.append((np.reshape(X_train[i], (128,1)), n_vec))

test_data = []
for i, n in enumerate(y_test):
    test_data.append((np.reshape(X_test[i], (128,1)), n))

dict_base_name = "training_"

# initialize a network for training
image_size = 128
num_of_hidden_neurons = 32
num_of_labels = 26
net = Network([image_size, num_of_hidden_neurons, num_of_labels])
# train NN:
num_of_epochs = 30
mini_batch_size = 10
eta = 1.0
net.SGD(training_data, num_of_epochs, mini_batch_size, eta, test_data)

# save results:
#base_path = "/home/ohadsh/Dropbox/Rami/Code_ohad/outputs/NN_basic"
#train_path = os.path.join(base_path, dict_base_name)
#os.makedirs(train_path)
#training_results_handler.save_net(net, os.path.join(train_path, "net.pcl"))
#training_results_handler.save_net_params(net.sizes, len(training_data),
#                                         len(test_data), num_of_epochs, mini_batch_size,
#                                     eta, net.test_results, os.path.join(train_path, "info.txt"))

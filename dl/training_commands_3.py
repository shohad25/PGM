import os, sys
import random
import cPickle
import numpy as np
from dl.src.network3 import Network
from dl.src.network3 import FullyConnectedLayer, SoftmaxLayer, ConvPoolLayer
from common.datasets.load_datasets import mnist_loader, load_letters
import theano

# Activation functions for neurons
def linear(z): return z
import theano.tensor as T
def ReLU(z): return T.maximum(0.0, z)
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh
from dl.user_network import training_results_handler

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

# Reshape to images
# X_train = X_train.reshape((-1, 8, 16))
# X_test = X_test.reshape((-1, 8, 16))

dict_base_name = "training_"

# initialize a network for training
image_size = 128
num_of_hidden_neurons = 128
num_of_labels = 26
mini_batch_size = np.int64(10)

# net = Network([image_size, 128, num_of_labels])
# conv_layer = ConvPoolLayer(filter_shape=(4, 1, 3, 3),
#                            image_shape=(mini_batch_size, 1, 16, 8), poolsize=(2, 2), activation_fn=ReLU)
fc_layer = FullyConnectedLayer(128, 256, activation_fn=sigmoid, p_dropout=0.2)
fc_layer1 = FullyConnectedLayer(256, 512, activation_fn=sigmoid, p_dropout=0.2)
fc_layer2 = FullyConnectedLayer(512, 256, activation_fn=sigmoid, p_dropout=0.2)
fc_layer3 = FullyConnectedLayer(256, 128, activation_fn=sigmoid, p_dropout=0.2)
sm_layer = SoftmaxLayer(128, 26)
# net = Network(layers=[conv_layer, fc_layer, fc_layer2, sm_layer], mini_batch_size=mini_batch_size)
# net = Network(layers=[fc_layer, fc_layer2, sm_layer], mini_batch_size=mini_batch_size)
net = Network(layers=[fc_layer, fc_layer1, fc_layer2, fc_layer3, sm_layer], mini_batch_size=mini_batch_size)
# train NN:
num_of_epochs = 1000

train_set_x = theano.shared(np.asarray(X_train, dtype=theano.config.floatX))
train_set_y = theano.shared(np.asarray(y_train, dtype=theano.config.floatX))
test_set_x = theano.shared(np.asarray(X_test, dtype=theano.config.floatX))
test_set_y = theano.shared(np.asarray(y_test, dtype=theano.config.floatX))

# train_set_x = theano.shared(value=X_train)
# test_set_x = theano.shared(value=X_test)
# test_set_x = theano.shared(np.asarray(X_test, dtype=theano.config.floatX))

train_set_y = theano.shared(np.asarray(y_train, dtype=theano.config.floatX))
test_set_y = theano.shared(np.asarray(y_test, dtype=theano.config.floatX))


train_set_y = theano.tensor.cast(train_set_y, 'int32')
test_set_y = theano.tensor.cast(test_set_y, 'int32')

eta = 0.1
net.SGD((train_set_x, train_set_y), num_of_epochs, mini_batch_size, eta,
        (test_set_x, test_set_y), (test_set_x, test_set_y))

# save results:
base_path = '/sheard/googleDrive/Master/courses/probabilistic_graphical_models/outputs/part_3/'
train_path = os.path.join(base_path, dict_base_name)
os.makedirs(train_path)
training_results_handler.save_net(net, os.path.join(train_path, "net.pcl"))
# training_results_handler.save_net_params(net.layers, len(X_train),
#                                         len(X_test), num_of_epochs, mini_batch_size,
#                                     eta, net.test_results, os.path.join(train_path, "info.txt"))

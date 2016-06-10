import os, sys
import random
import cPickle

from dl.src.network import Network
from common.datasets.load_datasets import mnist_loader
from dl.user_network import *
import pdb
# import rlcompleter
# pdb.Pdb.complete = rlcompleter.Completer(locals()).complete

# Training data - list of tuples: each tuple contains (a,b):
# a - numpy array - float 32 image (784)
# b - numpy array - float 64 of binary(float) 0.0,1.0 - (len = 10)
# Test & Validation data - list of tuples : 10,000. each tuple contains (a,b)
# a - # a - numpy array - float 32 image (784)
# b - int 64 - only one number (label 0-9)

# get the data
training_data, validation_data, test_data = mnist_loader()
# del training_data

# base_path = os.path.join(os.getcwd(), "data")
# f = open(os.path.join(base_path, 'canny_train.pkl'), 'r+')
# training_data = cPickle.load(f)
# # f.close()
#
#
# # convert_vec_to_num = lambda (vec): vec.
# f = open(os.path.join(base_path, 'canny_test.pkl'), 'r+')
# test_data = cPickle.load(f)
# f.close()

# e1 = training_data[0][0]
# e2 = training_data2[0][0]
# pdb.set_trace()

training_lengths = [50000, 40000, 30000, 20000, 10000, 5000, 1000, 500, 100, 10]

# training_lengths = [10]

dict_base_name = "training_"

# output directory:
# base_path = os.path.join(os.getcwd(), "output2")

training_index = 1
num_of_epochs = 30
mini_batch_size = 10
eta = 3.0
num_of_test_data = 10000

for length in training_lengths:

    # initialize a network and data for training
    net = Network([784, 30, 10])
    training_data_to_use = random.sample(training_data, length)

    # training
    print " Start of Training " + str(training_index)
    net.SGD(training_data_to_use, num_of_epochs, mini_batch_size, eta, test_data)
    print " End of Training " + str(training_index)

    # save results:
    base_path = "/home/ohadsh/Dropbox/Rami/Code_ohad/outputs/NN_basic"
    train_path = os.path.join(base_path, dict_base_name+str(training_index))
    os.makedirs(train_path)
    training_results_handler.save_net(net, os.path.join(train_path, "net.pcl"))
    training_results_handler.save_net_params(net.sizes, len(training_data_to_use),
                                         num_of_test_data, num_of_epochs, mini_batch_size,
                                         eta, net.test_results, os.path.join(train_path, "info.txt"))
    training_index += 1

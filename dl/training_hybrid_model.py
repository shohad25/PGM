"""
Train hybrid model: output of each network :
    basic, sobel, canny, harris, laplacian
    And combined them into new NN
"""
import os, sys
import cPickle
import random
from src.network import Network
from src import network as network
from src import mnist_loader
from user_network import *

# training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
# del training_data

basic_dir = os.getcwd()
training_data_dir = os.path.join(basic_dir, "data/nets_with_ftrs_norm.pkl")
f = open(training_data_dir, 'r+')
training_data_all = cPickle.load(f)
f.close()

test_data_dir = os.path.join(basic_dir, "data/nets_with_ftrs_test_norm.pkl")
f = open(test_data_dir, 'r+')
test_data = cPickle.load(f)
f.close()

training_lengths = [50000, 40000, 30000, 20000, 10000, 5000, 1000, 500, 100, 10]
# training_lengths = [5000]

# training_lengths = [10]

dict_base_name = "training_"

# output directory:
# base_path = os.path.join(os.getcwd(), "output2")

training_index = 1
num_of_epochs = 15
mini_batch_size = 10
eta = 3.0
num_of_test_data = 10000

for length in training_lengths:

    # initialize a network and data for training
    net = Network([50, 25, 10])
    training_data_to_use = training_data_all[length]
    test_data_to_use = test_data[length]
    # training
    print " Start of Training " + str(training_index)
    # net.SGD(training_data_to_use, num_of_epochs, mini_batch_size, eta, test_data)
    net.SGD(training_data_to_use, num_of_epochs, mini_batch_size, eta, test_data_to_use)
    print " End of Training " + str(training_index)

    # save results:
    base_path = "/home/ohadsh/Dropbox/Rami/Code_ohad/outputs/NN_hybrid_norm"
    train_path = os.path.join(base_path, dict_base_name+str(training_index))
    os.makedirs(train_path)
    training_results_handler.save_net(net, os.path.join(train_path, "net.pcl"))
    training_results_handler.save_net_params(net.sizes, len(training_data_to_use),
                                         num_of_test_data, num_of_epochs, mini_batch_size,
                                         eta, net.test_results, os.path.join(train_path, "info.txt"))
    training_index += 1



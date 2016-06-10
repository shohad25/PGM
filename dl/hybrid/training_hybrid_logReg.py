"""
Train hybrid model: output of each network :
    basic, sobel, canny, harris, laplacian
    And combined them into SVM
"""
import os
import cPickle
import sys
from hybrid.logistic_regreesion import *

sys.path.append('/home/ohadsh/Dropbox/Rami/Code_ohad')
from src import mnist_svm

basic_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
training_data_dir = os.path.join(basic_dir, "data/nets_with_ftrs_norm.pkl")
f = open(training_data_dir, 'r+')
training_data_all = cPickle.load(f)
f.close()

test_data_dir = os.path.join(basic_dir, "data/nets_with_ftrs_test_norm.pkl")
f = open(test_data_dir, 'r+')
test_data = cPickle.load(f)
f.close()

training_lengths = [50000, 40000, 30000, 20000, 10000, 5000, 1000, 500, 100, 10]

dict_base_name = "training_"

training_index = 1
num_of_test_data = 10000

training_results = {}

for length in training_lengths:

    training_data_to_use, test_data_to_use =\
        prepare_data_for_learners(training_data_all[length], test_data[length])

    print "LOG REG - Start of Training with " + str(length) + " Examples"
    rate = logistic_regression(training_data_to_use, test_data_to_use)
    print " End of Training " + str(length) + " Examples"
    # save results:
    training_results.setdefault(length, rate)
    training_index += 1

training_data_dir = os.path.join(basic_dir, "outputs/LogReg_hybrid/logreg.pkl")
f = open(training_data_dir, 'w+')
cPickle.dump(training_results, f)
f.close()
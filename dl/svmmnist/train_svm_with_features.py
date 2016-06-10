"""
Train hybrid model: output of each network :
    basic, sobel, canny, harris, laplacian
    And combined them into SVM
"""
import os
import cPickle
import sys
sys.path.append('/home/ohadsh/Dropbox/Rami/Code_ohad')
from hybrid.logistic_regreesion import prepare_data_for_learners
import math
import numpy as np
from src import mnist_svm2
import random
from sklearn import svm
from user_network import tools

# basic_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
basic_dir = '/home/ohadsh/Dropbox/Rami/Code_topaz/preprocessing/output'
training_data_dir = os.path.join(basic_dir, "train_features.pkl")
f = open(training_data_dir, 'r+')
training_data_all = cPickle.load(f)
f.close()

training_data_all[0] = training_data_all[0][:,[0,2,3,4,5,6,7,8,9,10]]
max_values = np.array([training_data_all[0][:,i].max() for i in range(0, 10)])
min_values = np.array([training_data_all[0][:,i].min() for i in range(0, 10)])
training_data_all[0] = training_data_all[0] / (max_values - min_values)

test_data_dir = os.path.join(basic_dir, "test_images_features.pkl")
f = open(test_data_dir, 'r+')
test_data = cPickle.load(f)
f.close()

test_data[0] = test_data[0][:,[0,2,3,4,5,6,7,8,9,10]]
# normalize:
max_values = np.array([test_data[0][:,i].max() for i in range(0, 10)])
min_values = np.array([test_data[0][:,i].min() for i in range(0, 10)])
test_data[0] = test_data[0] / (max_values - min_values)


# test_data[0] = not_non_vec2(test_data[0])

training_lengths = [50000, 40000, 30000, 20000, 10000, 5000, 1000, 500, 100, 10]

# training_lengths = [10]

dict_base_name = "training_"

training_index = 1
num_of_test_data = 10000

training_results = {}
predictions_results = {}
clf_results = {}
training_predictions_prob = {}

for length in training_lengths:
    training_data_chosen = random.sample(zip(*training_data_all), length)
    # import pdb
    # pdb.set_trace()
    training_data_to_use, test_data_to_use =\
        prepare_data_for_learners(training_data_chosen, zip(*test_data))

    print "SVM - Start of Training with " + str(length) + " Examples"
    rate, predictions, clf = mnist_svm2.svm_baseline(training_data_to_use, test_data_to_use)
    print " End of Training " + str(length) + " Examples"
    # save results:
    predictions_prob = [a for a in clf.predict_proba(test_data[0])]
    training_predictions_prob.setdefault(length, predictions_prob)
    training_results.setdefault(length, rate)
    predictions_results.setdefault(length, predictions)
    clf_results.setdefault(length, clf)
    training_index += 1

tools.dump_pkl(training_results, '/home/ohadsh/Dropbox/Rami/Code_ohad/outputs/SVM_features/svm_features_results.pkl')
tools.dump_pkl(predictions_results, '/home/ohadsh/Dropbox/Rami/Code_ohad/outputs/SVM_features/svm_features_predictions.pkl')
tools.dump_pkl(predictions_results, '/home/ohadsh/Dropbox/Rami/Code_ohad/outputs/SVM_features/svm_features_predictions_prob.pkl')
tools.dump_pkl(clf_results, '/home/ohadsh/Dropbox/Rami/Code_ohad/outputs/SVM_features/svm_features_clf.pkl')


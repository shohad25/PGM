#!/home/ohadsh/anaconda/bin/python
import sys, os
import random
#as
# sys.path.append(os.getcwd())
# parent_path = os.path.abspath(__file__ + "/../../")
# sys.path.append(parent_path)
sys.path.append('/home/ohadsh/Dropbox/Rami/Code_ohad')
from src import mnist_loader
from src import mnist_svm
import cPickle
from user_network import tools

training_data, validation_data, test_data = mnist_loader.load_data()


training_lengths = [50000, 40000, 30000, 20000, 10000, 5000, 1000, 500, 100, 10]
# training_lengths = [10]

training_results = {}
training_predictions_test = {}
training_predictions_prob = {}
training_predictions_train = {}
training_predictions_prob_train = {}
training_clf = {}

# output directory:
# parent_path = os.path.abspath(__file__ + "/../../")
parent_path = '/home/ohadsh/Dropbox/Rami/Code_ohad'
base_path = os.path.join(parent_path, "outputs/SVM_prob/")

training_index = 1
num_of_test_data = 10000

for length in training_lengths:

    training_data_to_use = random.sample(zip(*training_data), length)
    # training_data_to_use = training_data

    # import pdb
    # pdb.set_trace()

    # training
    print "SVM - Start of Training with " + str(length) + " Examples"
    rate, predictions, clf = mnist_svm.svm_baseline(zip(*training_data_to_use), test_data)
    print " End of Training " + str(length) + " Examples"

    # save results:
    predictions_prob = [a for a in clf.predict_proba(test_data[0])]
    predictions_prob_train = [a for a in clf.predict_proba((zip(*training_data_to_use))[0])]
    predictions_train = [a for a in clf.predict((zip(*training_data_to_use))[0])]

    training_results.setdefault(length, rate)
    training_predictions_test.setdefault(length, predictions)
    training_predictions_prob.setdefault(length, predictions_prob)

    training_predictions_train.setdefault(length, predictions_train)
    training_predictions_prob_train.setdefault(length, predictions_prob_train)

    training_clf.setdefault(length, clf)
    training_index += 1

tools.dump_pkl(training_results, os.path.join(base_path, 'svm_training_results.pkl'))
tools.dump_pkl(training_clf, os.path.join(base_path, 'svm_clf.pkl'))
tools.dump_pkl(training_predictions_test, os.path.join(base_path, 'svm_predictions.pkl'))
tools.dump_pkl(training_predictions_prob, os.path.join(base_path, 'svm_predictions_prob.pkl'))

tools.dump_pkl(training_predictions_train, os.path.join(base_path, 'svm_predictions_train.pkl'))
tools.dump_pkl(training_predictions_prob_train, os.path.join(base_path, 'svm_predictions_prob_train.pkl'))

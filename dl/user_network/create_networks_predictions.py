import os, sys
import random
import cPickle

from src.network import Network
from src import network as network
from src import mnist_loader
from user_network import analyzer

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
del validation_data

mode = 'Test'

# create analyzer
my_analyzer = analyzer.Analyzer()
# set training's data
basic_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

# list_of_dicts = ['sobel', 'laplacian', 'canny', 'harris', 'basic2']
list_of_dicts = ['basic2']
norm = lambda (x) : x / float(sum(x))

for dic_name in list_of_dicts:
    print "Start prediciotn on " + dic_name
    training_directory = os.path.join(basic_dir, "outputs/NN_" + dic_name)
    my_analyzer.update_trainings_results(training_directory)

    # compare_nets = lambda net : net.num_of_training_data
    nets = my_analyzer.nets
    predictions = {}
    for net in nets:
        if mode == 'Train':
            training_data_to_use = training_data_to_use = random.sample(training_data,
                                                                        net.num_of_training_data)
            training_results = [(norm(net.feedforward(x)), y) for (x, y) in training_data_to_use]
            predictions.setdefault(net.num_of_training_data, training_results)
        else:
            test_results = [(norm(net.feedforward(x)), y) for (x, y) in test_data]
            predictions.setdefault(net.num_of_training_data, test_results)

    if mode == 'Train':
        f = open(os.path.join(training_directory, 'predictions_norm.pkl'), 'w+')
    else:
        f = open(os.path.join(training_directory, 'predictions_test_norm.pkl'), 'w+')
    cPickle.dump(predictions, f)
    f.close()
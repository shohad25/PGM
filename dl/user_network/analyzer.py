"""
analyzer.py
~~~~~~~~~~~~~~~~~~~~~~~~

Includes several function which used for analyze the training results.
Contains function for comparing training's performance and plot the results.
"""
import os
import pickle
import matplotlib.pylab as plt
import numpy as np
import src.network as network


class Analyzer():
    def __init__(self, num_of_epoch=30, num_of_training_test=10000):
        """
        :param num_of_epoch: number of epochs - the same for all trainings
        :param num_of_training_test: evaluation set
        :return:
        """
        self.num_of_trainings = 0
        self.num_of_epoch = num_of_epoch
        self.num_of_training_test = num_of_training_test
        self.nets = []
        self.results = {}  # key : num of training data
        self.max_results = {}
        self.n_training_data_sorted = []

    def update_trainings_results(self, training_directory):
        """
        Collect all training results into self.trainings
        :param training_directory: Path to training results
        :return:
        """
        temp_dict = {}

        for dict_name in os.listdir(training_directory):
            if not dict_name.startswith('training_'):
                continue
            train_path = os.path.join(training_directory, dict_name)
            f = open(os.path.join(train_path, 'net.pcl'), 'r')
            net = pickle.load(f)
            self.nets.append(net)
            temp_dict.setdefault(net.num_of_training_data, net.test_results)
            f.close()

        self.results = temp_dict
        self.n_training_data_sorted = [net_iter.num_of_training_data for net_iter in self.nets]  # Ascending
        self.n_training_data_sorted.sort()

    def save_max_performance(self, name):
        self.max_results = {key: max(val) for (key, val) in self.results.iteritems()}
        f = open(name, 'w+')
        pickle.dump(self.max_results, f)
        f.close()

    def plot_result(self, n_training_data=[]):
        """
        :param n_training_data: Specific training results, if empty, plot all results
               for example plot_result([10, 10000, 30000]) - will plot 3 graph on the same figure
        :return:
        """
        x_axes = np.linspace(1, 30, 30, dtype='int')
        if not n_training_data:  # plot all results
            for res in self.n_training_data_sorted:
                plt.plot(x_axes, self.results[res])
            plt.legend(self.n_training_data_sorted, loc='upper left')

        #  Specific training data
        elif len(n_training_data) <= len(self.n_training_data_sorted):
            # Check input validity
            all_exists = not (False in [res in self.n_training_data_sorted for res in n_training_data])
            if all_exists:
                for res in n_training_data:
                    plt.plot(x_axes, self.results[res])
                plt.legend(n_training_data, loc='upper left')
            else:
                print " The provided training data isn't exists, please check the input"
                return

        plt.xlabel('Epochs')
        plt.ylabel('Classification rate')
        plt.show()

    def dict_to_vec(self, dic):
        return [dic[length] for length in sorted(dic.keys())]
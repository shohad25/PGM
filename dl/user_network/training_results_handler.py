"""
training_results_handler.py
~~~~~~~~~~~~~~~~~~~~~~~~

A module to save and load all relevant information of the training scheme.
Save the net objects using pickle and the training parameters at info.txt
Also, all epochs output saved in evaluation

"""

#### Libraries
# Standard library
import pickle

def save_net(net, file_name):
    """
    :param net: Nueral Network
    :param file_name: Output file - for saving the net
    :return: Status
    """
    if net and file_name:
        try:
            file_object = open(file_name, 'wb')
            pickle.dump(net, file_object)
            file_object.close()
            print "Saving data completed"
        except AttributeError:
            print "Can't save the data, please check the input"

def save_net_params(network_sizes, training_data_len, test_data_len, num_of_hidden_layers,
                    mini_batch_size, learning_rate, training_results, file_name):
    """
    :param net: Nueral Network
    :param file_name: Output file - for saving the network parameters
    :return: Status
    """
    try:
        file_object = open(file_name, 'wb')
        str_out = ('network_sizes = ' + str(network_sizes) + '\n ' +
                   'training_data_len = ' + str(training_data_len) + '\n ' +
                   'test_data_len = ' + str(test_data_len) + '\n ' +
                   'num_of_hidden_layers = ' + str(num_of_hidden_layers) + '\n ' +
                   'mini_batch_size = ' + str(mini_batch_size) + '\n ' +
                   'learning_rate = ' + str(learning_rate) + '\n ' +
                   'results = ' + str(training_results))
        file_object.write(str_out)
        file_object.close()
        print "Saving data completed"
    except AttributeError:
        print "Can't save the data, please check the input"


def load_net(file_name):
    """
    :rtype : network object
    :param file_name: Path of the net object
    :return: network object
    """
    try:
        file_object = open(file_name, "r")
        net = pickle.load(file_object)
        file_object.close()
        return net
    except AttributeError:
        print "Can't load the data, please check the pcl file"

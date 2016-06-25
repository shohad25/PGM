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


def predict(net, layer_ind=-1):
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

    # HACK - pad to 10 in order to predict all examples
    real_train = X_train.shape[0]
    real_test = X_test.shape[0]
    X_train = np.vstack((X_train, X_train[-6:-1]))
    X_test = np.vstack((X_test, X_test[-4:-1]))
    y_train = np.hstack((y_train, y_train[-6:-1]))
    y_test = np.hstack((y_test, y_test[-4:-1]))


    train_set_x = theano.shared(np.asarray(X_train, dtype=theano.config.floatX))
    train_set_y = theano.shared(np.asarray(y_train, dtype=theano.config.floatX))
    test_set_x = theano.shared(np.asarray(X_test, dtype=theano.config.floatX))
    test_set_y = theano.shared(np.asarray(y_test, dtype=theano.config.floatX))

    train_set_y = theano.shared(np.asarray(y_train, dtype=theano.config.floatX))
    test_set_y = theano.shared(np.asarray(y_test, dtype=theano.config.floatX))

    train_set_y = theano.tensor.cast(train_set_y, 'int32')
    test_set_y = theano.tensor.cast(test_set_y, 'int32')

    print "Predict Train:"
    predict_train = pred(net, train_set_x, train_set_y, layer_ind)
    print "Predict Test:"	
    predict_test = pred(net, test_set_x, test_set_y, layer_ind)

    return predict_train[0:real_train], predict_test[0:real_test]


def pred(net, data, labels, layer_ind):
        mini_batch_size = 10
        # compute number of minibatches for training, validation and testing
        num_batches = size(data)/mini_batch_size

        i = T.lscalar() # mini-batch index
        test_mb_accuracy = theano.function(
            [i], net.layers[-1].accuracy(net.y),
            givens={
                net.x: 
                data[i*mini_batch_size: (i+1)*mini_batch_size],
                net.y: 
                labels[i*mini_batch_size: (i+1)*mini_batch_size]
            })
        test_mb_predictions = theano.function(
            [i], net.layers[layer_ind].output,
            givens={
                net.x: 
                data[i*mini_batch_size: (i+1)*mini_batch_size]
            })
        # Do the actual prediction
        accuracy = np.mean(
            [test_mb_accuracy(j) for j in xrange(num_batches)])
        print('The corresponding accuracy is {0:.2%}'.format(
                            accuracy))
        prediction = [test_mb_predictions(j) for j in xrange(num_batches)]
        prediction = np.vstack(np.array(prediction))

        return prediction


#### Miscellanea
def size(data):
    "Return the size of the dataset `data`."
    return data.get_value(borrow=True).shape[0]

if __name__ == '__main__':
    net_path = '/media/ohadsh/sheard/googleDrive/Master/courses/probabilistic_graphical_models/outputs/part_3/training_2016_06_11/net.pcl'
    out_pred_path = '/media/ohadsh/sheard/googleDrive/Master/courses/probabilistic_graphical_models/outputs/part_3/training_2016_06_11/'
    # Load pre-trained network
    layer_ind = -1
    with open(net_path, 'r') as f:
        net = cPickle.load(f)
    pred_train, pred_test = predict(net, layer_ind=layer_ind)

    with open(os.path.join(out_pred_path, 'train_pred_%d.pkl' % layer_ind), 'w') as f:
        cPickle.dump(pred_train, f)

    with open(os.path.join(out_pred_path, 'test_pred_%d.pkl' % layer_ind), 'w') as f:
        cPickle.dump(pred_test, f)

    # for i in range(0, pred_train.shape[0]):
    #     max_ind = pred_train[i].argmax()
    #     pred_train[i].argmax()
    #     pred_train[i].fill(0)
    #     pred_train[i][max_ind] = 1
    #
    # for i in range(0, pred_test.shape[0]):
    #     max_ind = pred_test[i].argmax()
    #     pred_test[i].argmax()
    #     pred_test[i].fill(0)
    #     pred_test[i][max_ind] = 1
    #
    # with open(os.path.join(out_pred_path, 'train_pred_%d_normed.pkl' % layer_ind), 'w') as f:
    #     cPickle.dump(pred_train, f)
    #
    # with open(os.path.join(out_pred_path, 'test_pred_%d_normed.pkl' % layer_ind), 'w') as f:
    #     cPickle.dump(pred_test, f)




from sklearn import linear_model
import numpy as np

bin2int = lambda x: int(np.where(x==1)[0])
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

sigmoid_vec = np.vectorize(sigmoid)


def prepare_data_for_learners(training_data, test_data=None):
    #  Training:
    X = []
    Y = []
    for (x, y) in training_data:
        Y.append(bin2int(y))
        X.append(np.squeeze(x))
    tr = (np.array(X), np.array(Y))

    X = []
    Y = []
    for (x, y) in test_data:
        Y.append(y)
        X.append(np.squeeze(x))

    te = (np.array(X), np.array(Y))
    return tr, te


def logistic_regression(training_data, test_data, validation_data=None):
    clf = linear_model.LogisticRegression()
    clf.fit(training_data[0], training_data[1])
    # test

    coeffs = clf.coef_
    intercept = clf.intercept_

    predictions = [int(a) for a in clf.predict(test_data[0])]
    num_correct = sum(int(a == y) for a, y in zip(predictions, test_data[1]))

    # wrongs = [(x, y, a) for x, a, y in zip(test_data[0], predictions, test_data[1]) if int(a != y)]
    #
    # print wrongs[0][0]
    # print wrongs[0][1]
    # print wrongs[0][2]
    # pred = np.dot(coeffs, wrongs[0][0]) + intercept
    # print sigmoid_vec(pred)
    # print sum(sigmoid_vec(pred))

    print "Baseline classifier using an Logistic Regression."
    print "%s of %s values correct." % (num_correct, len(test_data[1]))
    return float(num_correct) / float(len(test_data[1]))

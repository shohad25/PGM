"""
mnist_svm2
~~~~~~~~~

A classifier program for recognizing handwritten digits from the MNIST
data set, using an SVM classifier."""

#### Libraries
# My libraries
import mnist_loader 

# Third-party libraries
from sklearn import svm

def svm_baseline(training_data, test_data, validation_data=None):
    # training_data, validation_data, test_data = mnist_loader.load_data()
    # train
    clf = svm.SVC(C=1.5157, gamma=0.05, kernel='rbf', probability=True)
    clf.fit(training_data[0], training_data[1])
    # test
    predictions = [int(a) for a in clf.predict(test_data[0])]
    num_correct = sum(int(a == y) for a, y in zip(predictions, test_data[1]))
    print "Baseline classifier using an SVM."
    print "%s of %s values correct." % (num_correct, len(test_data[1]))
    rate = float(num_correct) / float(len(test_data[1]))
    return rate, predictions, clf

# if __name__ == "__main__":
#     svm_baseline()

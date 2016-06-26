"""
===============================
OCR Letter sequence recognition
===============================
This example illustrates the use of a chain CRF for optical character
recognition. The example is taken from Taskar et al "Max-margin markov random
fields".

Each example consists of a handwritten word, that was presegmented into
characters.  Each character is represented as a 16x8 binary image. The task is
to classify the image into one of the 26 characters a-z. The first letter of
every word was ommited as it was capitalized and the task does only consider
small caps letters.

We compare classification using a standard linear SVM that classifies
each letter individually with a chain CRF that can exploit correlations
between neighboring letters (the correlation is particularly strong
as the same words are used during training and testing).

The first figures shows the segmented letters of four words from the test set.
In set are the ground truth (green), the prediction using SVM (blue) and the
prediction using a chain CRF (red).

The second figure shows the pairwise potentials learned by the chain CRF.
The strongest patterns are "y after l" and "n after i".

There are obvious extensions that both methods could benefit from, such as
window features or non-linear kernels. This example is more meant to give a
demonstration of the CRF than to show its superiority.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from common.viewers.imshow import imshow
from pystruct.datasets import load_letters
from pystruct.models import ChainCRF, GraphCRF
from pystruct.learners import FrankWolfeSSVM
from sklearn.linear_model import LinearRegression
from common.utils import get_letters_in_pred_like, arrange_letters_in_pred_like
import cPickle
abc = "abcdefghijklmnopqrstuvwxyz"

# Load data:
letters = load_letters()
X, y, folds = letters['data'], letters['labels'], letters['folds']

# we convert the lists to object arrays, as that makes slicing much more
# convenient
X, y = np.array(X), np.array(y)
X_train, X_test = X[folds == 1], X[folds != 1]
y_train, y_test = y[folds == 1], y[folds != 1]


net_base_path = '/media/ohadsh/sheard/googleDrive/Master/courses/probabilistic_graphical_models/outputs/part_3/training_2016_06_11/'
# Load pre-trained network
train_name = 'train_pred_-2.pkl'
test_name = 'test_pred_-2.pkl'
with open(os.path.join(net_base_path, train_name), 'r') as f:
    train_net_pred = cPickle.load(f)
with open(os.path.join(net_base_path, test_name), 'r') as f:
    test_net_pred = cPickle.load(f)

# Rearrange data for CRF
nn_predictions_train = arrange_letters_in_pred_like(X_train, train_net_pred, size_of_pred=26)
nn_predictions_test = arrange_letters_in_pred_like(X_test, test_net_pred, size_of_pred=26)

# Train LCCRF
chain_model = ChainCRF(directed=True)
chain_ssvm = FrankWolfeSSVM(model=chain_model, C=.1, max_iter=11)
chain_ssvm.fit(X_train, y_train)

# Train LCCRF+NN
chain_model = ChainCRF(directed=True)
chain_ssvm_nn = FrankWolfeSSVM(model=chain_model, C=.1, max_iter=11)
chain_ssvm_nn.fit(nn_predictions_train, y_train)

print("Test score with linear NN: 84.15%")

print("Test score with LCCRF: %f" % chain_ssvm.score(X_test, y_test))

print("Test score with LCCRF+NN: %f" % chain_ssvm_nn.score(nn_predictions_test, y_test))

# plot some word sequenced
n_words = 4
rnd = np.random.RandomState(1)
selected = rnd.randint(len(y_test), size=n_words)
max_word_len = max([len(y_) for y_ in y_test[selected]])
fig, axes = plt.subplots(n_words, max_word_len, figsize=(10, 10))
fig.subplots_adjust(wspace=0)
fig.text(0.2, 0.05, 'GT', color="#00AA00", size=25)
fig.text(0.4, 0.05, 'NN', color="#5555FF", size=25)
fig.text(0.6, 0.05, 'LCCRF', color="#FF5555", size=25)
fig.text(0.8, 0.05, 'LCCRF+NN', color="#FFD700", size=25)

fig.text(0.05, 0.5, 'Word', color="#000000", size=25)
fig.text(0.5, 0.95, 'Letters', color="#000000", size=25)

with open(os.path.join(net_base_path, 'test_pred_-1_normed.pkl'), 'r') as f:
    test_net_pred_last = cPickle.load(f)
test_net_pred_last = arrange_letters_in_pred_like(X_test, test_net_pred_last, size_of_pred=26)

for ind, axes_row in zip(selected, axes):
    y_pred_nn = test_net_pred_last[ind].argmax(axis=1)
    y_pred_chain = chain_ssvm.predict([X_test[ind]])[0]
    y_pred_chain_nn = chain_ssvm_nn.predict([nn_predictions_test[ind]])[0]

    for i, (a, image, y_true, y_nn, y_chain, y_chain_nn) in enumerate(
            zip(axes_row, X_test[ind], y_test[ind], y_pred_nn, y_pred_chain, y_pred_chain_nn)):
        a.matshow(image.reshape(16, 8), cmap=plt.cm.Greys)
        a.text(0, 3, abc[y_true], color="#00AA00", size=25)    # Green
        a.text(0, 14, abc[y_nn], color="#5555FF", size=25)    # Blue
        a.text(5, 14, abc[y_chain], color="#FF5555", size=25)  # Red
        a.text(5, 3, abc[y_chain_nn], color="#FFD700", size=25)     # Yellow
        a.set_xticks(())
        a.set_yticks(())
    for ii in range(i + 1, max_word_len):
        axes_row[ii].set_visible(False)

w = chain_ssvm_nn.w[26 * 128:].reshape(26, 26)
w_prob = np.exp(w) / sum(np.exp(w))

fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].set_title('Transition parameters of LCCRF+NN.', fontsize=30)

plt.sca(ax[0])
plt.xticks(np.arange(26), abc, fontsize=20)
plt.yticks(np.arange(26), abc, fontsize=20)
imshow(w, ax=ax[0], fig=fig, colormap='rainbow')

ax[1].set_title('Transition Probability of LCCRF+NN.', fontsize=30)
plt.sca(ax[1])
plt.xticks(np.arange(26), abc, fontsize=20)
plt.yticks(np.arange(26), abc, fontsize=20)
imshow(w_prob, ax=ax[1], fig=fig, colormap='rainbow', block=True)



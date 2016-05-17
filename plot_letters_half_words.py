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

from sklearn.svm import LinearSVC
from common.viewers.imshow import imshow
from pystruct.datasets import load_letters
from pystruct.models import ChainCRF, GraphCRF
from pystruct.learners import FrankWolfeSSVM
from mpl_toolkits.axes_grid1 import make_axes_locatable

abc = "abcdefghijklmnopqrstuvwxyz"

# Load data:
letters = load_letters()
X, y, folds = letters['data'], letters['labels'], letters['folds']

# we convert the lists to object arrays, as that makes slicing much more
# convenient
X, y = np.array(X), np.array(y)
X_train, X_test = X[folds == 1], X[folds != 1]
y_train, y_test = y[folds == 1], y[folds != 1]

# Train linear SVM
svm = LinearSVC(dual=False, C=.1)
# flatten input

svm.fit(np.vstack(X_train), np.hstack(y_train))

# Train directed chain CRF
model = ChainCRF(directed=True)
ssvm = FrankWolfeSSVM(model=model, C=.1, max_iter=11)
ssvm.fit(X_train, y_train)


# Train undirected chain CRF
half_model = ChainCRF(directed=True)
half_ssvm = FrankWolfeSSVM(model=half_model, C=.1, max_iter=11, verbose=0)

# Creaete a database with "half" word
X_half_train = np.ones_like(np.concatenate((y_train, y_train)))
y_half_train = np.ones_like(X_half_train)

for ind in range(0, X_train.shape[0]):
    # n_letters = 2 #fixed len of word
    n_letters = int(np.floor(X_train[ind].shape[0] / 2))
    X_half_train[2*ind] = X_train[ind][0:n_letters]
    X_half_train[2*ind+1] = X_train[ind][n_letters:]
    y_half_train[2*ind] = y_train[ind][0:n_letters]
    y_half_train[2*ind+1] = y_train[ind][n_letters:]
# Train the model
half_ssvm.fit(X_half_train, y_half_train)


print("Test score with linear SVM: %f" % svm.score(np.vstack(X_test),
                                                   np.hstack(y_test)))
print("Test score with FULL LCCRF: %f" % ssvm.score(X_test, y_test))

print("Test score with HALF LCCRF: %f" % half_ssvm.score(X_test, y_test))

# plot some word sequenced
n_words = 4
rnd = np.random.RandomState(1)
selected = rnd.randint(len(y_test), size=n_words)
max_word_len = max([len(y_) for y_ in y_test[selected]])
fig, axes = plt.subplots(n_words, max_word_len, figsize=(10, 10))
fig.subplots_adjust(wspace=0)
fig.text(0.2, 0.05, 'GT', color="#00AA00", size=25)
fig.text(0.4, 0.05, 'SVM', color="#5555FF", size=25)
fig.text(0.6, 0.05, 'HALF-LCCRF', color="#FF5555", size=25)
fig.text(0.8, 0.05, 'FULL-LCCRF', color="#FFD700", size=25)

fig.text(0.05, 0.5, 'Word', color="#000000", size=25)
fig.text(0.5, 0.95, 'Letters', color="#000000", size=25)

for ind, axes_row in zip(selected, axes):
    y_pred_svm = svm.predict(X_test[ind])
    y_pred_half = half_ssvm.predict([X_test[ind]])[0]
    y_pred_crf = ssvm.predict([X_test[ind]])[0]

    for i, (a, image, y_true, y_svm, y_half, y_crf) in enumerate(
            zip(axes_row, X_test[ind], y_test[ind], y_pred_svm, y_pred_half, y_pred_crf)):
        a.matshow(image.reshape(16, 8), cmap=plt.cm.Greys)
        a.text(0, 3, abc[y_true], color="#00AA00", size=25)    # Green
        a.text(0, 14, abc[y_svm], color="#5555FF", size=25)    # Blue
        a.text(5, 14, abc[y_half], color="#FF5555", size=25)  # Red
        a.text(5, 3, abc[y_crf], color="#FFD700", size=25)     # Yellow
        a.set_xticks(())
        a.set_yticks(())
    for ii in range(i + 1, max_word_len):
        axes_row[ii].set_visible(False)

w = ssvm.w[26 * 8 * 16:].reshape(26, 26)
w_prob = np.exp(w) / sum(np.exp(w))


fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].set_title('Transition parameters of the full LCCRF.', fontsize=30)

plt.sca(ax[0])
plt.xticks(np.arange(26), abc, fontsize=20)
plt.yticks(np.arange(26), abc, fontsize=20)
imshow(w, ax=ax[0], fig=fig, colormap='rainbow')

ax[1].set_title('Transition Probability of the full LCCRF.', fontsize=30)
plt.sca(ax[1])
plt.xticks(np.arange(26), abc, fontsize=20)
plt.yticks(np.arange(26), abc, fontsize=20)
imshow(w_prob, ax=ax[1], fig=fig, colormap='rainbow', block=True)


w_half = half_ssvm.w[26 * 8 * 16:].reshape(26, 26)
w_half_prob = np.exp(w_half) / sum(np.exp(w_half))

fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].set_title('Transition parameters of the half LCCRF.', fontsize=30)

plt.sca(ax[0])
plt.xticks(np.arange(26), abc, fontsize=20)
plt.yticks(np.arange(26), abc, fontsize=20)
imshow(w_half, ax=ax[0], fig=fig, colormap='rainbow')

ax[1].set_title('Transition Probability of the half LCCRF.', fontsize=30)
plt.sca(ax[1])
plt.xticks(np.arange(26), abc, fontsize=20)
plt.yticks(np.arange(26), abc, fontsize=20)
imshow(w_half_prob, ax=ax[1], fig=fig, colormap='rainbow', block=True)
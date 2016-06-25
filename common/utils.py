import copy
import numpy as np


def get_letters_in_pred_like(data, predictor, size_of_pred):
    """
    Get predictions data as [0,0,0,0,0,1,0,0]
    :param data: Train/Test data (array of arrays)
    :param predictor: predictor - classifier
    :return: np array
    """
    pred_like = copy.deepcopy(data)
    for word_ind in range(0, len(pred_like)):
        word = predictor.predict(data[word_ind]).reshape((-1, 1))
        pred = np.zeros((len(word), 26))
        for i, w in enumerate(word):
            pred[i][int(w)] = 1.0
        pred_like[word_ind] = pred
    return pred_like


def arrange_letters_in_pred_like(data, predict, size_of_pred):
    """
    Get predictions data as [0,0,0,0,0,1,0,0]
    :param data: Train/Test data (array of arrays)
    :param predictor: predict from other classifier
    :return: np array
    """
    pred_like = copy.deepcopy(data)
    counter = 0
    for word_ind in range(0, len(pred_like)):

        word_len = data[word_ind].shape[0]
        word = predict[counter: counter + word_len]
        counter += word_len
        pred_like[word_ind] = word
    return pred_like

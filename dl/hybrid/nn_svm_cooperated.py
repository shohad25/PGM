"""
NN basic and SVM hybrid method: When NN's prediction have low confidence, use SVM prediction
 if it's better.
"""
import os
import cPickle
import sys
sys.path.append('/home/ohadsh/Dropbox/Rami/Code_ohad')
from hybrid.logistic_regreesion import prepare_data_for_learners
from src import mnist_svm
from user_network import tools
import numpy as np
from matplotlib import pyplot as plt

# basic_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
basic_dir = '/home/ohadsh/Dropbox/Rami/Code_ohad'
training_data_NN_path = os.path.join(basic_dir, "outputs/NN_basic2/predictions_norm.pkl")
predictions_NN_train = tools.load_pkl(training_data_NN_path)

test_data_NN_path = os.path.join(basic_dir, "outputs/NN_basic2/predictions_test_norm.pkl")
predictions_NN_test = tools.load_pkl(test_data_NN_path)

test_data_SVM_path = os.path.join(basic_dir, "outputs/SVM_prob/svm_predictions.pkl")
predictions_SVM_test = tools.load_pkl(test_data_SVM_path)
predictions_SVM_train = tools.load_pkl(os.path.join(basic_dir, "outputs/SVM_prob/svm_predictions_train.pkl"))

training_lengths = [50000, 40000, 30000, 20000, 10000, 5000, 1000, 500, 100, 10]

# training_lengths = [10]
bin2int = lambda x: int(np.where(x==1)[0])
dict_base_name = "training_"

training_index = 1
num_of_test_data = 10000

training_results = {}
all_data_mistakes = {}
all_data_good = {}
rates = {}
zero_one_scores = {}
thr_knots = np.linspace(0.0, 1.0, 100)

for length in training_lengths:

    nn_pred_digits_with_conf = [(pred_vec.max(), list(pred_vec).index(pred_vec.max()), g_t)
                     for pred_vec, g_t in predictions_NN_train[length]]
    nn_pred_mistakes_conf = [pred_conf for pred_conf, pred_digit, g_t in nn_pred_digits_with_conf
                             if int(pred_digit) != bin2int(g_t)]
    nn_pred_goods_conf = [pred_conf for pred_conf, pred_digit, g_t in nn_pred_digits_with_conf
                             if int(pred_digit) == bin2int(g_t)]

    all_data_mistakes.setdefault(length, nn_pred_mistakes_conf)
    all_data_good.setdefault(length, nn_pred_goods_conf)

    # plt.figure()
    # plt.hist(nn_pred_mistakes_conf, bins=100, normed=True, cumulative=True)
    # plt.title('Mistakes confidences CDF - basic NN : #Training set = ' + str(length))
    # plt.xlabel('confidence')
    # plt.figure()
    # plt.hist(nn_pred_goods_conf, bins=100, normed=True, cumulative=True)
    # plt.title('True classification :confidences CDF - basic NN : #Training set = ' + str(length))
    # plt.xlabel('confidence')
    ground_truth = np.array([bin2int(g_t) for pred_vec, g_t in predictions_NN_train[length]])
    svm_pred = predictions_SVM_train[length]
    precision_all = []
    recall_all = []
    f_score_all = []
    for thr in thr_knots:
        # nn_pred = np.array([pred_digit if pred_conf >= thr else -1
        #                     for pred_conf, pred_digit, g_t in nn_pred_digits_with_conf])
        # num_of_available = float(sum(nn_pred != -1))
        # availability = num_of_available / float(len(nn_pred))
        # tp = availability * (sum(nn_pred == ground_truth) / float(len(nn_pred)))
        # # fp = availability * (sum(nn_pred != ground_truth) / float(len(nn_pred)))
        # fp = sum(nn_pred != ground_truth) / float(len(nn_pred))
        # tn = 0.0
        # fn = 1 - availability
        # precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        # recall = tp / (tp + fn) if fp + fn > 0 else 0.0
        # f_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
        # print ('thr:' + str(thr) + ' availability:'+str(availability) + " " + 'tp:'+ str(tp) + " " + 'fp:'+ str(fp) + " " + 'fn:'+ str(fn)
        #        + " " + 'precision:'+ str(precision) + " " + 'recall:' + str(recall) + " " + 'f_score:' + str(f_score))
        # precision_all.append(precision)
        # recall_all.append(recall)
        # f_score_all.append(f_score)

        nn_pred = np.array([pred_digit if pred_conf >= thr else svm_p
                            for ((pred_conf, pred_digit, g_t), svm_p) in zip(nn_pred_digits_with_conf, svm_pred)])
        tp = (sum(nn_pred == ground_truth) / float(len(nn_pred)))
        fp = (sum(nn_pred != ground_truth) / float(len(nn_pred)))
        tn = 0.0
        fn = 0.0
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if fp + fn > 0 else 0.0
        f_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
        # print ('thr:' + str(thr) + " " + 'tp:'+ str(tp) + " " + 'fp:'+ str(fp) + " " + 'fn:'+ str(fn)
        #        + " " + 'precision:'+ str(precision) + " " + 'recall:' + str(recall) + " " + 'f_score:' + str(f_score))
        precision_all.append(precision)
        recall_all.append(recall)
        f_score_all.append(f_score)

    zero_one_scores.setdefault(length, dict(f_score=f_score_all, recall=recall_all, precision=precision_all))


    print length
    # find best thr:
    svm_pred_test = predictions_SVM_test[length]
    thr_opt = thr_knots[np.array(f_score_all).argmax()]
    nn_pred_digits_with_conf_test = [(pred_vec.max(), list(pred_vec).index(pred_vec.max()), g_t)
                     for pred_vec, g_t in predictions_NN_test[length]]

    pred_opt = np.array([pred_digit if pred_conf >= thr_opt else svm_p
                        for ((pred_conf, pred_digit, g_t), svm_p) in zip(nn_pred_digits_with_conf_test, svm_pred_test)])
    nn_pred = np.array([pred_digit for pred_conf, pred_digit, g_t in nn_pred_digits_with_conf_test])
    ground_truth_test = np.array([g_t for pred_vec, g_t in predictions_NN_test[length]])
    print "# training set = " + str(length)
    print 'NN classification rate:'
    nn_rate = sum(np.array(nn_pred) == ground_truth_test)
    print nn_rate
    print 'SVM classification rate:'
    svm_rate = sum(np.array(svm_pred_test) == ground_truth_test)
    print svm_rate
    print 'Hybrid classification rate:'
    rate = sum(np.array(pred_opt) == ground_truth_test)
    print rate
    rates.setdefault(length, {'NN': nn_rate, 'SVM': svm_rate, 'hybrid': rate})


# plt.show()
#
#
# for length in training_lengths:
#     if length == 10:
#         continue
#     f = zero_one_scores[length]['f_score']
#     plt.plot(thr_knots, f, label=length)
#     plt.xlabel('confidence')
#     plt.ylabel('f1_score')
#     plt.title("f1_score as function of confidence's threshold")
#
# plt.legend(loc=3)
# plt.show()
# #
#
#

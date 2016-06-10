import os, sys
import random
import cPickle

from src.network import Network
from src import network as network
from src import mnist_loader
from user_network import analyzer
import matplotlib.pyplot as plt

mode = 1

# create analyzer
my_analyzer = analyzer.Analyzer()
# set training's data
# training_directory = os.path.join(os.getcwd(), "outputs/LogReg_hybrid/")
# my_analyzer.update_trainings_results(training_directory)
# my_analyzer.save_max_performance(os.path.join(training_directory, 'logreg_res.pkl'))


# plot results
# train_tests = [30000, 20000, 50000]  # number of training data, if empty, will plot all results (10)
# my_analyzer.plot_result(train_tests)
if mode == 1:
    # NN basic
    f = open('/home/ohadsh/Dropbox/Rami/Code_ohad/outputs/NN_basic/nn_basic_res.pcl', 'r')
    nn_basic = cPickle.load(f)
    f.close()
    nn_vec = my_analyzer.dict_to_vec(nn_basic)

    # SVM basic
    f = open('/home/ohadsh/Dropbox/Rami/Code_ohad/outputs/SVM/svm_basic_res.pcl', 'r')
    svm_basic = cPickle.load(f)
    f.close()
    svm_vec = my_analyzer.dict_to_vec(svm_basic)

    # NN more hidden layer
    f = open('/home/ohadsh/Dropbox/Rami/Code_ohad/outputs/NN_MoreHiddenLayer/nn_moreHiddenLayer.pcl', 'r')
    nn_more_hidden_layer = cPickle.load(f)
    f.close()
    nn_more_hidden_layer_vec = my_analyzer.dict_to_vec(nn_more_hidden_layer)

    # NN shorter hidden layer
    f = open('/home/ohadsh/Dropbox/Rami/Code_ohad/outputs/NN_shorterHiddenLayer/nn_shorterHiddenLayer.pcl', 'r')
    nn_shorter_hidden_layer = cPickle.load(f)
    f.close()
    nn_shorter_hidden_layer_vec = my_analyzer.dict_to_vec(nn_shorter_hidden_layer)

    # NN canny
    f = open('/home/ohadsh/Dropbox/Rami/Code_ohad/outputs/NN_canny/nn_canny.pcl', 'r')
    nn_canny = cPickle.load(f)
    f.close()
    nn_canny_vec = my_analyzer.dict_to_vec(nn_canny)

    # NN harris
    f = open('/home/ohadsh/Dropbox/Rami/Code_ohad/outputs/NN_harris/nn_harris.pcl', 'r')
    nn_harris = cPickle.load(f)
    f.close()
    nn_harris_vec = my_analyzer.dict_to_vec(nn_harris)

    # NN sobel
    f = open('/home/ohadsh/Dropbox/Rami/Code_ohad/outputs/NN_sobel/nn_sobel.pcl', 'r')
    nn_sobel = cPickle.load(f)
    f.close()
    nn_sobel_vec = my_analyzer.dict_to_vec(nn_sobel)

    # NN laplacian
    f = open('/home/ohadsh/Dropbox/Rami/Code_ohad/outputs/NN_laplacian/nn_laplacian.pcl', 'r')
    nn_laplacian = cPickle.load(f)
    f.close()
    nn_laplacian_vec = my_analyzer.dict_to_vec(nn_laplacian)

    # NN noisy bg
    f = open('/home/ohadsh/Dropbox/Rami/Code_ohad/outputs/NN_noisy_bg/nn_noisy_bg.pcl', 'r')
    nn_noisy_bg = cPickle.load(f)
    f.close()
    nn_noisy_bg_vec = my_analyzer.dict_to_vec(nn_noisy_bg)

    # NN hybrid
    f = open('/home/ohadsh/Dropbox/Rami/Code_ohad/outputs/NN_hybrid/nn_hybrid.pcl', 'r')
    nn_hybrid = cPickle.load(f)
    f.close()
    nn_hybrid_vec = my_analyzer.dict_to_vec(nn_hybrid)

    # NN hybrid_norm
    f = open('/home/ohadsh/Dropbox/Rami/Code_ohad/outputs/NN_hybrid_norm/nn_hybrid_norm.pcl', 'r')
    nn_hybrid_norm = cPickle.load(f)
    f.close()
    nn_hybrid_norm_vec = my_analyzer.dict_to_vec(nn_hybrid_norm)

    # log reg hybrid_norm
    f = open('/home/ohadsh/Dropbox/Rami/Code_ohad/outputs/LogReg_hybrid/logreg.pkl', 'r')
    log_reg_hybrid = cPickle.load(f)
    f.close()
    log_reg_hybrid_vec = my_analyzer.dict_to_vec(log_reg_hybrid)

    # log reg hybrid_norm
    f = open('/home/ohadsh/Dropbox/Rami/Code_ohad/outputs/SVM_hybrid/svm_hybrid.pkl', 'r')
    svm_hybrid = cPickle.load(f)
    f.close()
    svm_hybrid_vec = my_analyzer.dict_to_vec(svm_hybrid)

    # NN_SVM_Hybrid
    f = open('/home/ohadsh/Dropbox/Rami/Code_ohad/outputs/NN_SVM_hybrid/rates.pkl', 'r')
    nn_svm_hybrid_rates = cPickle.load(f)
    f.close()
    nn_svm_hybrid_vec = [nn_svm_hybrid_rates[length]['hybrid'] / float(10000) for length in sorted(nn_svm_hybrid_rates.keys())]
    print nn_svm_hybrid_vec
    lengths = sorted(nn_basic.keys())

    plt.plot(lengths, nn_vec, 'r', label='NN')
    plt.plot(lengths, svm_vec, 'b', label='SVM')
    # plt.plot(lengths, nn_more_hidden_layer_vec, 'g', label='NN_moreHiddenLayer')
    # plt.plot(lengths, nn_shorter_hidden_layer_vec, 'm', label='NN_shorterHiddenLayer')
    # plt.plot(lengths, nn_canny_vec, 'r--', label='NN_canny')
    # plt.plot(lengths, nn_harris_vec, 'b--', label='NN_harris')
    # plt.plot(lengths, nn_sobel_vec, 'g--', label='NN_sobel')
    # plt.plot(lengths, nn_laplacian_vec, 'm--', label='NN_laplacian')
    # plt.plot(lengths, nn_noisy_bg_vec, 'c', label='NN_noisy_bg')
    # plt.plot(lengths, nn_hybrid_vec, 'm--', label='NN_hybrid')
    plt.plot(lengths, nn_hybrid_norm_vec, 'r--', label='NN_hybrid_norm')
    plt.plot(lengths, log_reg_hybrid_vec, 'b--', label='log_reg_hybrid')
    plt.plot(lengths, svm_hybrid_vec, 'g--', label='svm_hybrid')
    # plt.plot(lengths, nn_svm_hybrid_vec, 'm--', label='NN_SVM_Hybrid')
    # plt.legend(bbox_to_anchor=(1.0, 0.5))
    plt.legend(loc=4)
    plt.xlabel('# Training examples')
    plt.ylabel('Max classification rate')
    plt.title('NN_SVM_Hybrid')
    plt.show()

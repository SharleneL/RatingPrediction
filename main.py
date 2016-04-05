__author__ = 'luoshalin'

import sys
import numpy as np
from preprocess import get_review_list, print_analysis
from featureExtract import gen_libsvm_file
from featureExtract import get_train_feature_M
from featureExtract import get_test_feature_M
from lr import lr_train_param
from lr import lr_predict
from evaluate import cal_accuracy
from evaluate import cal_rmse
from evaluate import print_pred_res

# change file paths
# change for i in range(10):  # for testing 10->2000 in featureExtract.py


def main(argv):
    # ====== / PARAMS / ====== #
    # FILE PATH PARAMS
    # input_fpath = sys.argv[1]
    # output_fpath = sys.argv[2]
    # ******************** / test|real - start / ******************** #
    # input_fpath = '../../resources/data/small/yelp_reviews_train_small_10000.json'  # test
    input_fpath = '../../resources/data/yelp_reviews_train.json'  # real - train set

    # test_input_fpath = '../../resources/data/yelp_reviews_dev.json'  # real - test set

    test_input_fpath = '../../resources/data/yelp_reviews_test.json'  # real - test set
    # ******************** / test|real - end / ******************** #
    output_fpath = 'pred_result.out'
    stopword_fpath = '../../resources/data/stopword.list'
    # FEATURE EXTRACTION PARAMS
    feature_method = 'ctf'
    # feature_method = 'df'
    # ******************** / test|real - start / ******************** #
    # feature_num = 10        # test
    feature_num = 4000    # real
    # ******************** / test|real - end / ******************** #
    class_num = 5
    # LR PARAMS
    lmd = 0.01
    alpha = 0.001  # learning rate
    # threshold = 10E-6  # stopping criteria - bsga
    threshold = 10E-6  # stopping criteria
    # gd_method = 'sga'
    gd_method = 'bsga'
    # ******************** / test|real - start / ******************** #
    # batch_size = 10  # test
    batch_size = 200  # real
    # ******************** / test|real - end / ******************** #
    eval_data_frac = 0.04   # how much data to be splitted out as evaluation data


    print 'START!'
    # ========== / generate files - COMMENTED IF FILES ARE ALREADY GENERATED - START / ========== #
    # TASK1: PREPROCESS - each row of stars_M is a vector.T of corresponding review. only one col in the vector is '1'
    # review_list, stars_list, total_data_amt, test_review_list, test_stars_list, test_total_data_amt = get_review_list(stopword_fpath, input_fpath, test_input_fpath)  # stars_M: <datanum, classnum:5>
    # print "Train - total data amount: " + str(total_data_amt)
    # print "Test - total data amount: " + str(test_total_data_amt)
    # # print_analysis(review_list)
    # print 'TASK1 - end'
    #
    # # TASK2: FEATURE DESIGN
    # # generate review_feature & feature_dict files
    # libsvm_file_path, test_libsvm_file_path = gen_libsvm_file(review_list, stars_list, test_review_list, feature_method, feature_num)  # get the file for useful features
    # print libsvm_file_path
    # print test_libsvm_file_path
    # ========== / generate files - COMMENTED IF FILES ARE ALREADY GENERATED - END/ ========== #


    # HARD CODED PARTS
    # === file path === #
    # libsvm_file_path = '../feature_data/ctf_libsvm.out'  # hard coded filepath - for efficiency
    # test_libsvm_file_path = '../feature_data/test/ctf_libsvm_test_dev.out'  # hard coded filepath - for efficiency
    libsvm_file_path = '../feature_data/test/4000/ctf_libsvm_train.out'  # hard coded filepath - for efficiency
    test_libsvm_file_path = '../feature_data/test/4000/ctf_libsvm_test.out'  # hard coded filepath - for efficiency
    # === data amount === #
    # total_data_amt = 1255353        # hard coded total data amount - train
    # total_data_amt = 157010        # hard coded total data amount - dev
    total_data_amt = 156901        # hard coded total data amount - test

    # generate features
    train_M, eval_M, train_stars_M, eval_stars_M, eval_stars_list = get_train_feature_M(libsvm_file_path, total_data_amt, eval_data_frac, feature_num, class_num)
    test_M = get_test_feature_M(test_libsvm_file_path, feature_num)

    # TASK3.1: LR TRAIN
    W_org = np.ones(shape=(feature_num, class_num)) * float(1)/feature_num
    W = lr_train_param(train_M, train_stars_M, W_org, eval_M, eval_stars_M, lmd, alpha, threshold, gd_method, batch_size)  # eval_M is the evaluation dataset

    # TASK3.2: EVALUATION
    # on the evaluation set
    pred_list_hard, pred_list_soft = lr_predict(eval_M, W)
    # on the test/dev set
    test_pred_list_hard, test_pred_list_soft = lr_predict(test_M, W)
    # output test/dev evaluation results to file
    print_pred_res(test_pred_list_hard, test_pred_list_soft, output_fpath)

    # calculate accuracy
    accuracy = cal_accuracy(eval_stars_list, pred_list_hard)
    print 'Accuracy: ' + str(accuracy)
    # calculate rmse
    rmse = cal_rmse(eval_stars_list, pred_list_soft)
    print 'RMSE: ' + str(rmse)

if __name__ == '__main__':
    main(sys.argv[1:])
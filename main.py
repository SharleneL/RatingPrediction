__author__ = 'luoshalin'

import sys
import numpy as np
from preprocess import get_review_list, print_analysis
from featureExtract import gen_libsvm_file
from featureExtract import get_feature_M
from lr import lr_train_param
from lr import lr_predict
from evaluate import cal_accuracy

# change file paths
# change for i in range(10):  # for testing 10->2000 in featureExtract.py


def main(argv):
    # ====== / PARAMS / ====== #
    # FILE PATH PARAMS
    # input_fpath = sys.argv[1]
    # output_fpath = sys.argv[2]
    # ******************** / test|real - start / ******************** #
    input_fpath = '../../resources/data/small/yelp_reviews_train_small_10000.json'  # test
    # input_fpath = '../../resources/data/yelp_reviews_train.json'  # real
    # ******************** / test|real - end / ******************** #
    # output_fpath = '../../resources/data/small/yelp_reviews_train_output.txt'
    stopword_fpath = '../../resources/data/stopword.list'
    # FEATURE EXTRACTION PARAMS
    feature_method = 'ctf'
    # feature_method = 'df'
    # ******************** / test|real - start / ******************** #
    # feature_num = 10        # test
    feature_num = 2000    # real
    # ******************** / test|real - end / ******************** #
    class_num = 5
    # LR PARAMS
    lmd = 0.01
    alpha = 0.001  # learning rate
    threshold = 10E-7  # stopping criteria
    gd_method = 'sga'
    # gd_method = 'bsga'
    # ******************** / test|real - start / ******************** #
    # batch_size = 10  # test
    batch_size = 100  # real
    # ******************** / test|real - end / ******************** #
    pred_method = 'hard'
    # pred_method = 'soft'
    eval_data_frac = 0.04   # how much data to be splitted out as evaluation data


    print 'START!'
    # ========== / generate files - COMMENTED IF FILES ARE ALREADY GENERATED - START / ========== #
    # # TASK1: PREPROCESS - each row of stars_M is a vector.T of corresponding review. only one col in the vector is '1'
    # review_list, stars_list, total_data_amt = get_review_list(stopword_fpath, input_fpath, class_num, eval_data_frac)  # stars_M: <datanum, classnum:5>
    # print total_data_amt
    # # print_analysis(review_list)
    # print 'TASK1 - end'
    #
    # # TASK2: FEATURE DESIGN
    # # generate review_feature & feature_dict files
    # libsvm_file_path = gen_libsvm_file(review_list, stars_list, feature_method, feature_num)  # get the file for useful features
    # print libsvm_file_path
    # ========== / generate files - COMMENTED IF FILES ARE ALREADY GENERATED - END/ ========== #


    libsvm_file_path = '../feature_data/small_ctf_libsvm.out'  # hard coded filepath - for efficiency
    # total_data_amt = 1255353        # hard coded total data amount - real
    total_data_amt = 10000        # hard coded total data amount - test
    # generate features
    train_M, eval_M, train_stars_M, eval_stars_M, eval_stars_list = get_feature_M(libsvm_file_path, total_data_amt, eval_data_frac, feature_num, class_num)
    print 'TASK2 - eval_M got & end'
    # print train_M.shape
    # print eval_M.shape
    # print train_stars_M.shape
    # print eval_stars_M.shape

    # print train_M.toarray()[0][7]
    # print train_M.toarray()[0][773]
    # print train_M.toarray()[8][40]
    # print train_M.toarray()[8][0]
    # print train_M.toarray()[7][40]
    # print train_M.toarray()[7][1955]
    # print train_M.toarray()[9599][240]
    # print eval_M.toarray()[0][1571]  # line 9601
    # print eval_M.toarray()[0][914]
    # print eval_M.toarray()[0][48]
    # print eval_M.toarray()[2][962]
    # print eval_M.toarray()[399][142]
    # print eval_stars_M[0]
    # print eval_stars_M[1]
    # print eval_stars_M[2]
    # print eval_stars_M[3]

    # TASK3.1: MODEL DESIGN - LR
    # TRAIN
    W_org = np.ones(shape=(feature_num, class_num)) * float(1)/feature_num
    print 'TASK3 - W_org got'
    W = lr_train_param(train_M, train_stars_M, W_org, eval_M, eval_stars_M, lmd, alpha, threshold, gd_method, batch_size)  # eval_M is the evaluation dataset
    print 'TASK3 - W got'
    # print W
    # TEST on the evaluation set
    pred_list = lr_predict(eval_M, W, pred_method)
    print 'TASK3 - pred_list got'
    # for p in pred_list:
    #     print p
    # calculate accuracy
    accuracy = cal_accuracy(eval_stars_list, pred_list)
    print accuracy

    # TASK3.2: MODEL DESIGN - SVM
    # TRAIN

    # TEST


if __name__ == '__main__':
    main(sys.argv[1:])